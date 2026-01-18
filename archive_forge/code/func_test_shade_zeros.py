from __future__ import annotations
from io import BytesIO
import numpy as np
import xarray as xr
import dask.array as da
import PIL
import pytest
import datashader.transfer_functions as tf
from datashader.tests.test_pandas import assert_eq_ndarray, assert_eq_xr, assert_image_close
@pytest.mark.parametrize('array', arrays)
def test_shade_zeros(array):
    coords = [np.array([0, 1]), np.array([2, 5])]
    cat_agg = tf.Image(array([[(0, 0, 0), (0, 0, 0)], [(0, 0, 0), (0, 0, 0)]], dtype='u4'), coords=coords + [['a', 'b', 'c']], dims=dims + ['cats'])
    colors = [(255, 0, 0), '#0000FF', 'orange']
    img = tf.shade(cat_agg, color_key=colors, how='linear', min_alpha=0)
    sol = np.array([[5584810, 5584810], [5584810, 5584810]], dtype='u4')
    sol = tf.Image(sol, coords=coords, dims=dims)
    assert_eq_xr(img, sol)