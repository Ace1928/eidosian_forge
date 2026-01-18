from __future__ import annotations
from io import BytesIO
import numpy as np
import xarray as xr
import dask.array as da
import PIL
import pytest
import datashader.transfer_functions as tf
from datashader.tests.test_pandas import assert_eq_ndarray, assert_eq_xr, assert_image_close
def test_rgb_spread():
    p = 2097152125
    g = 2097217280
    b = 2113863680
    data = np.array([[p, p, 0, 0, 0], [p, g, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, b, 0], [0, 0, 0, 0, 0]], dtype='uint32')
    coords = [np.arange(5), np.arange(5)]
    img = tf.Image(data, coords=coords, dims=dims)
    s = tf.spread(img)
    o = np.array([[3976234555, 3976234555, 3154159658, 0, 0], [3976234555, 3976234555, 3154159658, 0, 0], [3154159658, 3154159658, 3165148672, 2113863680, 2113863680], [0, 0, 2113863680, 2113863680, 2113863680], [0, 0, 2113863680, 2113863680, 2113863680]])
    np.testing.assert_equal(s.data, o)
    assert (s.x_axis == img.x_axis).all()
    assert (s.y_axis == img.y_axis).all()
    assert s.dims == img.dims
    s = tf.spread(img, px=2)
    o = np.array([[3976234555, 3976234555, 3976234555, 3154159658, 0], [3976234555, 3976234555, 4118888732, 3700443154, 2113863680], [3976234555, 4118888732, 3984999449, 3165148672, 2113863680], [3154159658, 3700443154, 3165148672, 2113863680, 2113863680], [0, 2113863680, 2113863680, 2113863680, 2113863680]])
    np.testing.assert_equal(s.data, o)
    s = tf.spread(img, shape='square')
    o = np.array([[3976234555, 3976234555, 3154159658, 0, 0], [3976234555, 3976234555, 3154159658, 0, 0], [3154159658, 3154159658, 3165148672, 2113863680, 2113863680], [0, 0, 2113863680, 2113863680, 2113863680], [0, 0, 2113863680, 2113863680, 2113863680]])
    np.testing.assert_equal(s.data, o)
    s = tf.spread(img, how='add')
    o = np.array([[4278222263, 4278222263, 4194336574, 0, 0], [4278222263, 4278222263, 4194336574, 0, 0], [4194336574, 4194336574, 4202659584, 2113863680, 2113863680], [0, 0, 2113863680, 2113863680, 2113863680], [0, 0, 2113863680, 2113863680, 2113863680]])
    np.testing.assert_equal(s.data, o)
    mask = np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]])
    s = tf.spread(img, mask=mask)
    o = np.array([[3154159658, 3154116733, 2097217280, 0, 0], [3154116733, 3154159658, 2097152125, 0, 0], [2097217280, 2097152125, 3165148672, 0, 2113863680], [0, 0, 0, 2113863680, 0], [0, 0, 2113863680, 0, 2113863680]])
    np.testing.assert_equal(s.data, o)
    s = tf.spread(img, px=0)
    np.testing.assert_equal(s.data, img.data)
    pytest.raises(ValueError, lambda: tf.spread(img, px=-1))
    pytest.raises(ValueError, lambda: tf.spread(img, mask=np.ones(2)))
    pytest.raises(ValueError, lambda: tf.spread(img, mask=np.ones((2, 2))))