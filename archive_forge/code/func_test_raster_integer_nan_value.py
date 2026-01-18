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
def test_raster_integer_nan_value():
    """
    Ensure custom nan_value is handled correctly for integer arrays.
    """
    cvs = ds.Canvas(plot_height=2, plot_width=2, x_range=(0, 1), y_range=(0, 1))
    array = np.array([[9999, 1, 2, 3], [4, 9999, 6, 7], [8, 9, 9999, 11]])
    coords = {'x': np.linspace(0, 1, 4), 'y': np.linspace(0, 1, 3)}
    xr_array = xr.DataArray(array, coords=coords, dims=['y', 'x'])
    agg = cvs.raster(xr_array, downsample_method='max', nan_value=9999)
    expected = np.array([[4, 7], [9, 11]])
    assert np.allclose(agg.data, expected)
    assert agg.data.dtype.kind == 'i'
    assert np.allclose(agg.x.values, np.array([0.25, 0.75]))
    assert np.allclose(agg.y.values, np.array([0.25, 0.75]))