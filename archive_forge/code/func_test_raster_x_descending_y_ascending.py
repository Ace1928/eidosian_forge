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
def test_raster_x_descending_y_ascending():
    """
    Assert raster with descending x- and ascending y-coordinates is aggregated correctly.
    """
    xs = np.arange(10)[::-1]
    ys = np.arange(5)
    arr = xs * ys[np.newaxis].T
    xarr = xr.DataArray(arr, coords={'X': xs, 'Y': ys}, dims=['Y', 'X'])
    cvs = ds.Canvas(10, 5, x_range=(-0.5, 9.5), y_range=(-0.5, 4.5))
    agg = cvs.raster(xarr)
    assert np.allclose(agg.data, arr)
    assert np.allclose(agg.X.values, xs)
    assert np.allclose(agg.Y.values, ys)
    assert np.allclose(agg.x_range, (-0.5, 9.5))
    assert np.allclose(agg.y_range, (-0.5, 4.5))