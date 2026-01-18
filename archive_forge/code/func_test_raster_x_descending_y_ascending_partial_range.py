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
def test_raster_x_descending_y_ascending_partial_range():
    """
    Assert raster with descending x- and ascending y-coordinates is aggregated correctly.
    """
    xs = np.arange(10)[::-1]
    ys = np.arange(5)
    arr = xs * ys[np.newaxis].T
    xarr = xr.DataArray(arr, coords={'X': xs, 'Y': ys}, dims=['Y', 'X'])
    cvs = ds.Canvas(7, 2, x_range=(0.5, 7.5), y_range=(1.5, 3.5))
    agg = cvs.raster(xarr)
    assert np.allclose(agg.data, xarr.sel(X=slice(7, 1), Y=slice(2, 3)).data)
    assert np.allclose(agg.X.values, xs[2:9])
    assert np.allclose(agg.Y.values, ys[2:4])