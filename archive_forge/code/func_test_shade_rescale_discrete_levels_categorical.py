from __future__ import annotations
from io import BytesIO
import numpy as np
import xarray as xr
import dask.array as da
import PIL
import pytest
import datashader.transfer_functions as tf
from datashader.tests.test_pandas import assert_eq_ndarray, assert_eq_xr, assert_image_close
@pytest.mark.parametrize('array_module', array_modules)
def test_shade_rescale_discrete_levels_categorical(array_module):
    arr = array_module.array([[[1, 2], [0, 1]], [[0, 0], [0, 0]], [[1, 0], [3, 0]], [[1, 0], [2, 1]]], dtype='u4')
    agg = xr.DataArray(data=arr, coords=dict(y=[0, 1, 2, 3], x=[0, 1], cat=['a', 'b']))
    img = tf.shade(agg, how='eq_hist', rescale_discrete_levels=True)
    sol = np.array([[4286864496, 1874361911], [6966413, 6966413], [1864112868, 4280031972], [1864112868, 4283448234]])
    assert_eq_ndarray(img.data, sol)