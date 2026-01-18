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
def test_shade_category(array):
    coords = [np.array([0, 1]), np.array([2, 5])]
    cat_agg = tf.Image(array([[(0, 12, 0), (3, 0, 3)], [(12, 12, 12), (24, 0, 0)]], dtype='u4'), coords=coords + [['a', 'b', 'c']], dims=dims + ['cats'])
    colors = [(255, 0, 0), '#0000FF', 'orange']
    img = tf.shade(cat_agg, color_key=colors, how='log', min_alpha=20)
    sol = np.array([[2583625728, 335565567], [4283774890, 3707764991]], dtype='u4')
    sol = tf.Image(sol, coords=coords, dims=dims)
    assert_eq_xr(img, sol)
    assert list(img.coords) == ['x_axis', 'y_axis']
    assert list(img.dims) == ['y_axis', 'x_axis']
    colors = dict(zip('abc', colors))
    img = tf.shade(cat_agg, color_key=colors, how='cbrt', min_alpha=20)
    sol = np.array([[2650734592, 335565567], [4283774890, 3657433343]], dtype='u4')
    sol = tf.Image(sol, coords=coords, dims=dims)
    assert_eq_xr(img, sol)
    img = tf.shade(cat_agg, color_key=colors, how='linear', min_alpha=20)
    sol = np.array([[1140785152, 335565567], [4283774890, 2701132031]], dtype='u4')
    sol = tf.Image(sol, coords=coords, dims=dims)
    assert_eq_xr(img, sol)
    img = tf.shade(cat_agg, color_key=colors, how=lambda x, m: np.where(m, np.nan, x) ** 2, min_alpha=20)
    sol = np.array([[503250944, 335565567], [4283774890, 1744830719]], dtype='u4')
    sol = tf.Image(sol, coords=coords, dims=dims)
    assert_eq_xr(img, sol)
    img = tf.shade(cat_agg, color_key=colors, how='linear', min_alpha=0, span=(50, 100))
    sol = np.array([[16711680, 21247], [5584810, 255]], dtype='u4')
    sol = tf.Image(sol, coords=coords, dims=dims)
    assert_eq_xr(img, sol)
    assert img.data[0, 0] >> 24 & 255 == 0
    assert img.data[0, 1] >> 24 & 255 == 0
    assert img.data[1, 0] >> 24 & 255 == 0
    assert img.data[1, 1] >> 24 & 255 == 0
    img = tf.shade(cat_agg, color_key=colors, how='linear', min_alpha=0, span=(0, 2))
    sol = np.array([[4294901760, 4278211327], [4283774890, 4278190335]], dtype='u4')
    sol = tf.Image(sol, coords=coords, dims=dims)
    assert_eq_xr(img, sol)
    assert img.data[0, 0] >> 24 & 255 == 255
    assert img.data[0, 1] >> 24 & 255 == 255
    assert img.data[1, 0] >> 24 & 255 == 255
    assert img.data[1, 1] >> 24 & 255 == 255
    img = tf.shade(cat_agg, color_key=colors, how='linear', min_alpha=0, span=(6, 36))
    sol = np.array([[872349696, 21247], [4283774890, 2566914303]], dtype='u4')
    sol = tf.Image(sol, coords=coords, dims=dims)
    assert_eq_xr(img, sol)
    assert img.data[0, 0] >> 24 & 255 == 51
    assert img.data[0, 1] >> 24 & 255 == 0
    assert img.data[1, 0] >> 24 & 255 == 255
    assert img.data[1, 1] >> 24 & 255 == 153
    img = tf.shade(cat_agg, color_key=colors, how='linear', min_alpha=0, span=(0, 72))
    sol = np.array([[721354752, 352342783], [2136291242, 1426063615]], dtype='u4')
    sol = tf.Image(sol, coords=coords, dims=dims)
    assert_eq_xr(img, sol)
    assert img.data[0, 0] >> 24 & 255 == 42
    assert img.data[0, 1] >> 24 & 255 == 21
    assert img.data[1, 0] >> 24 & 255 == 127
    assert img.data[1, 1] >> 24 & 255 == 85
    cat_agg = tf.Image(array([[(0, 0, 0), (3, 0, 3)], [(12, 12, 12), (24, 0, 0)]], dtype='u4'), coords=coords + [['a', 'b', 'c']], dims=dims + ['cats'])
    img = tf.shade(cat_agg, color_key=colors, how='linear', min_alpha=20)
    sol = np.array([[5584810, 335565567], [4283774890, 2701132031]], dtype='u4')
    sol = tf.Image(sol, coords=coords, dims=dims)
    assert_eq_xr(img, sol)
    assert img.data[0, 0] >> 24 & 255 == 0
    assert img.data[0, 1] >> 24 & 255 != 0
    assert img.data[1, 0] >> 24 & 255 != 0
    assert img.data[1, 1] >> 24 & 255 != 0
    img = tf.shade(cat_agg, color_key=colors, how='linear', min_alpha=20, span=(6, 36))
    sol = np.array([[5584810, 335565567], [4283774890, 2701132031]], dtype='u4')
    sol = tf.Image(sol, coords=coords, dims=dims)
    assert_eq_xr(img, sol)
    assert img.data[0, 0] >> 24 & 255 == 0
    assert img.data[0, 1] >> 24 & 255 != 0
    assert img.data[1, 0] >> 24 & 255 != 0
    assert img.data[1, 1] >> 24 & 255 != 0
    cat_agg = tf.Image(array([[(0, -30, 0), (18, 0, -18)], [(-2, 2, -2), (-18, 9, 12)]], dtype='i4'), coords=coords + [['a', 'b', 'c']], dims=dims + ['cats'])
    img = tf.shade(cat_agg, color_key=colors, how='linear', min_alpha=20)
    sol = np.array([[335565567, 3914667690], [3680253090, 4285155988]], dtype='u4')
    sol = tf.Image(sol, coords=coords, dims=dims)
    assert_eq_xr(img, sol)
    assert img.data[0, 0] >> 24 & 255 == 20
    assert img.data[0, 1] >> 24 & 255 == 233
    assert img.data[1, 0] >> 24 & 255 == 219
    assert img.data[1, 1] >> 24 & 255 == 255
    img = tf.shade(cat_agg, color_key=colors, how='linear', min_alpha=20, span=(0, 3))
    sol = np.array([[335565567, 341120682], [341587106, 4285155988]], dtype='u4')
    sol = tf.Image(sol, coords=coords, dims=dims)
    assert_eq_xr(img, sol)
    assert img.data[0, 0] >> 24 & 255 == 20
    assert img.data[0, 1] >> 24 & 255 == 20
    assert img.data[1, 0] >> 24 & 255 == 20
    assert img.data[1, 1] >> 24 & 255 == 255
    img = tf.shade(cat_agg, color_key=colors, how='linear', min_alpha=20, color_baseline=9)
    sol = np.array([[341129130, 3909091583], [3679795114, 4278232575]], dtype='u4')
    sol = tf.Image(sol, coords=coords, dims=dims)
    assert_eq_xr(img, sol)
    assert img.data[0, 0] >> 24 & 255 == 20
    assert img.data[0, 1] >> 24 & 255 == 233
    assert img.data[1, 0] >> 24 & 255 == 219
    assert img.data[1, 1] >> 24 & 255 == 255
    cat_agg = tf.Image(array([[(0, -30, 0), (-18, 0, -18)], [(-2, -2, -2), (-18, 0, 0)]], dtype='i4'), coords=coords + [['a', 'b', 'c']], dims=dims + ['cats'])
    img = tf.shade(cat_agg, color_key=colors, how='linear', min_alpha=20)
    sol = np.array([[1124094719, 344794225], [4283774890, 2708096148]], dtype='u4')
    sol = tf.Image(sol, coords=coords, dims=dims)
    assert_eq_xr(img, sol)
    assert img.data[0, 0] >> 24 & 255 == 67
    assert img.data[0, 1] >> 24 & 255 == 20
    assert img.data[1, 0] >> 24 & 255 == 255
    assert img.data[1, 1] >> 24 & 255 == 161
    img = tf.shade(cat_agg, color_key=colors, how='linear', min_alpha=20, span=(6, 36))
    sol = np.array([[335565567, 344794225], [341129130, 342508692]], dtype='u4')
    sol = tf.Image(sol, coords=coords, dims=dims)
    assert_eq_xr(img, sol)
    assert img.data[0, 0] >> 24 & 255 == 20
    assert img.data[0, 1] >> 24 & 255 == 20
    assert img.data[1, 0] >> 24 & 255 == 20
    assert img.data[1, 1] >> 24 & 255 == 20