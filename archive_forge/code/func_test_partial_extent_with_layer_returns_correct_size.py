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
@open_rasterio_available
def test_partial_extent_with_layer_returns_correct_size(cvs):
    with open_rasterio(TEST_RASTER_PATH) as src:
        res = ds.utils.calc_res(src)
        left, bottom, right, top = ds.utils.calc_bbox(src.x.values, src.y.values, res)
        half_width = (right - left) / 2
        half_height = (top - bottom) / 2
        cvs = ds.Canvas(plot_width=512, plot_height=256, x_range=[left - half_width, left + half_width], y_range=[bottom - half_height, bottom + half_height])
        agg = cvs.raster(src, layer=1)
        assert agg.shape == (256, 512)
        assert agg is not None