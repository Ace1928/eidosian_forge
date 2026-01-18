from __future__ import annotations
from io import BytesIO
import numpy as np
import xarray as xr
import dask.array as da
import PIL
import pytest
import datashader.transfer_functions as tf
from datashader.tests.test_pandas import assert_eq_ndarray, assert_eq_xr, assert_image_close
def test_rgb_dynspread():
    b = 4294901760
    coords = [np.arange(5), np.arange(5)]
    data = np.array([[b, b, 0, 0, 0], [b, b, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, b, 0], [0, 0, 0, 0, 0]], dtype='uint32')
    img = tf.Image(data, coords=coords, dims=dims)
    assert tf.dynspread(img).equals(img)
    data = np.array([[b, 0, 0, 0, 0], [0, 0, 0, 0, 0], [b, 0, 0, 0, b], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]], dtype='uint32')
    img = tf.Image(data, coords=coords, dims=dims)
    assert tf.dynspread(img, threshold=0.4).equals(tf.spread(img, 0))
    assert tf.dynspread(img, threshold=0.7).equals(tf.spread(img, 1))
    assert tf.dynspread(img, threshold=1.0).equals(tf.spread(img, 3))
    assert tf.dynspread(img, max_px=0).equals(img)
    pytest.raises(ValueError, lambda: tf.dynspread(img, threshold=1.1))
    pytest.raises(ValueError, lambda: tf.dynspread(img, max_px=-1))