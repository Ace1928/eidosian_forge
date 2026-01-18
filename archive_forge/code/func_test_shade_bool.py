from __future__ import annotations
from io import BytesIO
import numpy as np
import xarray as xr
import dask.array as da
import PIL
import pytest
import datashader.transfer_functions as tf
from datashader.tests.test_pandas import assert_eq_ndarray, assert_eq_xr, assert_image_close
def test_shade_bool():
    data = ~np.eye(3, dtype='bool')
    x = tf.Image(data, coords=coords, dims=dims)
    sol = tf.Image(np.where(data, 4278190335, 0).astype('uint32'), coords=coords, dims=dims)
    img = tf.shade(x, cmap=['pink', 'red'], how='log')
    assert_eq_xr(img, sol)
    img = tf.shade(x, cmap=['pink', 'red'], how='cbrt')
    assert_eq_xr(img, sol)
    img = tf.shade(x, cmap=['pink', 'red'], how='linear')
    assert_eq_xr(img, sol)
    img = tf.shade(x, cmap=['pink', 'red'], how='eq_hist')
    assert_eq_xr(img, sol)