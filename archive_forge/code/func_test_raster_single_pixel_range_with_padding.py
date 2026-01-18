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
def test_raster_single_pixel_range_with_padding():
    """
    Ensure that canvas range covering a single pixel and small area
    beyond the defined data ranges is handled correctly.
    """
    cvs = ds.Canvas(plot_height=4, plot_width=6, x_range=(-0.5, 0.25), y_range=(-0.5, 0.301))
    cvs2 = ds.Canvas(plot_height=4, plot_width=6, x_range=(-0.5, 0.25), y_range=(-0.5, 0.3))
    array = np.array([[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]], dtype='f')
    xr_array = xr.DataArray(array, dims=['y', 'x'], coords={'x': np.linspace(0.125, 0.875, 4), 'y': np.linspace(0.125, 0.625, 3)})
    agg = cvs.raster(xr_array, downsample_method='max', nan_value=np.nan)
    agg2 = cvs2.raster(xr_array, downsample_method='max', nan_value=np.nan)
    expected = np.array([[np.nan, np.nan, np.nan, np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan, np.nan, 0, 0], [np.nan, np.nan, np.nan, np.nan, 0, 0]])
    expected2 = np.array([[np.nan, np.nan, np.nan, np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan, np.nan, 0, 0]])
    assert np.allclose(agg.data, expected, equal_nan=True)
    assert np.allclose(agg2.data, expected2, equal_nan=True)
    assert agg.data.dtype.kind == 'f'
    assert np.allclose(agg.x.values, np.array([-0.4375, -0.3125, -0.1875, -0.0625, 0.0625, 0.1875]))
    assert np.allclose(agg.y.values, np.array([-0.399875, -0.199625, 0.000625, 0.200875]))