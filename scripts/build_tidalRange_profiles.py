import logging
import time
from itertools import product

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
from atlite.gis import ExclusionContainer

from scripts._helpers import (
    configure_logging,
    get_snapshots,
    load_cutout,
    set_scenario_config,
    setup_dask,
)
from scripts.build_shapes import _simplify_polys

logger = logging.getLogger(__name__)


if __name__ == "__main__":
    if "snakemake" not in globals():
        from scripts._helpers import mock_snakemake

        snakemake = mock_snakemake(
            "build_tidalRange_profiles", clusters=38, technology="tidalRange"
        )
    configure_logging(snakemake)
    set_scenario_config(snakemake)

    nprocesses = int(snakemake.threads)
    noprogress = snakemake.config["run"].get("disable_progressbar", True)
    noprogress = noprogress or not snakemake.config["atlite"]["show_progress"]
    technology = "tidalRange"
    params = snakemake.params.renewable[technology]
    resource = params["resource"]  
    resource["show_progress"] = not noprogress
    dask_kwargs = setup_dask(nprocesses)
    sns = get_snapshots(snakemake.params.snapshots, snakemake.params.drop_leap_day)
    cutout = load_cutout(snakemake.input.cutout, time=sns)
    correction_factor = params.get("correction_factor", 1.0)

    availability = xr.open_dataarray(snakemake.input.availability_matrix)
    pe_min = params["pe_min"]
    
    pe_per_sqkm = cutout.tidalrange()
    pe_filter = (pe_per_sqkm>pe_min).astype(int).values
    availability = availability * pe_filter  

    regions = gpd.read_file(snakemake.input.distance_regions)
    # do not pull up, set_index does not work if geo dataframe is empty
    regions = regions.set_index("name").rename_axis("bus")

    # for offshore regions, the shortest distance to the shoreline is used
    offshore_regions = availability.coords["bus"].values
    regions = regions.loc[offshore_regions]
    regions = regions.map(lambda g: _simplify_polys(g, minarea=1)).set_crs(
        regions.crs
    )

    regions = regions.geometry.to_crs(3035)

    area = cutout.grid.to_crs(3035).area / 1e6
    area = xr.DataArray(
        area.values.reshape(cutout.shape), [cutout.coords["y"], cutout.coords["x"]]
    )

    func = getattr(cutout, resource.pop("method"))

    # indicator matrix for which cells touch which regions
    kwargs = dict(nprocesses=nprocesses, disable_progressbar=noprogress)
    capacity_per_sqkm = cutout.tidalrange(return_capacity=True)
    layout =  (area * capacity_per_sqkm).compute()
    mask = (availability.sum(axis=0)!= 0).astype(int).values
    logger.info(f"Calculate capacity factor time series")
    start = time.time()
    matrix = (availability).stack(spatial=["y", "x"])
    profile = func(
        matrix=matrix,
        layout=layout,
        index=matrix.indexes["bus"],
        per_unit=True,
        return_capacity=False,
        mask=mask,
        dask_kwargs=dask_kwargs,
        **resource,
    )

    duration = time.time() - start
    logger.info(
        f"Completed weighted capacity factor time series calculation ({duration:2.2f}s)"
    )

    logger.info(f"Calculating maximal capacity per bus for technology {technology}")
    p_nom_max =  capacity_per_sqkm * availability @ area

    logger.info(f"Calculate average distances for technology {technology}.")

    # layout: (y, x)
    # availability: (bus, y, x)
    # Result: (bus, spatial), where spatial = (y, x)
    layoutmatrix = (layout * availability).stack(spatial=["y", "x"])

    coords = cutout.grid.representative_point().to_crs(3035)

    average_distance = []
    for bus in availability.coords["bus"].values:
        row = layoutmatrix.sel(bus=bus).data
        nz_b = row != 0
        row = row[nz_b]
        co = coords[nz_b]
        distances = co.distance(regions[bus]).div(1e3)
        average_distance.append(
            (distances * (row / row.sum())).sum()
        )

    average_distance = xr.DataArray(
        average_distance,
        coords={"bus": availability.coords["bus"]},
        dims=["bus"],
    )
    print("ending")

    ds = xr.merge(
        [
            (correction_factor * profile).rename("profile"),
            p_nom_max.rename("p_nom_max"),
            average_distance.rename("average_distance"),
        ]
    )

    # Select only buses with sufficient profile and potential capacity
    mean_profile = ds["profile"].mean("time")
    sum_potential = ds["p_nom_max"]

    ds = ds.sel(
        bus=(
            (mean_profile > params.get("min_p_max_pu", 0.0))
            & (sum_potential > params.get("min_p_nom_max", 0.0))
        )
    )

    # Clip low capacity-factor values if requested
    if "clip_p_max_pu" in params:
        min_p_max_pu = params["clip_p_max_pu"]
        ds["profile"] = ds["profile"].where(
            ds["profile"] >= min_p_max_pu, 0
        )

    ds.to_netcdf(snakemake.output.profile)