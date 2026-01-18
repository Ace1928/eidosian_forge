from __future__ import annotations
import pytest
from dask.context import config
from os import path
from itertools import product
import datashader as ds
import xarray as xr
import numpy as np
import dask.array as da
import pandas as pd
from datashader.resampling import compute_chunksize
import datashader.transfer_functions as tf
from packaging.version import Version
def test_raster_distributed_regrid_chunksize():
    """
    Ensure that distributed regrid respects explicit chunk size.
    """
    cvs = ds.Canvas(plot_height=2, plot_width=2)
    size = 4
    vs = np.linspace(-1, 1, size)
    xs, ys = np.meshgrid(vs, vs)
    arr = np.sin(xs * ys)
    darr = da.from_array(arr, (2, 2))
    xr_darr = xr.DataArray(darr, coords=[('y', range(size)), ('x', range(size))], name='z')
    agg_darr = cvs.raster(xr_darr, chunksize=(1, 1))
    assert agg_darr.data.chunksize == (1, 1)