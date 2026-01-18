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
def test_raster_vs_points_coords():
    points = pd.DataFrame(data=dict(x=[2, 6, 8], y=[9, 7, 3]))
    raster = xr.DataArray(data=[[0.0, 1.0], [2.0, 3.0]], dims=('y', 'x'), coords=dict(x=[0, 9], y=[0, 11]))
    canvas = ds.Canvas(25, 15, x_range=(0, 10), y_range=(0, 5))
    agg_points = canvas.points(points, x='x', y='y')
    agg_raster = canvas.raster(raster)
    im_points = tf.shade(agg_points)
    im_raster = tf.shade(agg_raster)
    np.testing.assert_array_equal(im_points.coords['x'], im_raster.coords['x'])
    np.testing.assert_array_equal(im_points.coords['y'], im_raster.coords['y'])