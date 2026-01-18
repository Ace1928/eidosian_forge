from __future__ import annotations
from io import BytesIO
import numpy as np
import xarray as xr
import dask.array as da
import PIL
import pytest
import datashader.transfer_functions as tf
from datashader.tests.test_pandas import assert_eq_ndarray, assert_eq_xr, assert_image_close
def test_shade_with_discrete_color_key():
    data = np.array([[0, 0, 0, 0, 0], [0, 1, 1, 1, 0], [0, 2, 2, 2, 0], [0, 3, 3, 3, 0], [0, 0, 0, 0, 0]], dtype='uint32')
    color_key = {1: 'white', 2: 'purple', 3: 'yellow'}
    result = np.array([[0, 0, 0, 0, 0], [0, 4294967295, 4294967295, 4294967295, 0], [0, 4286578816, 4286578816, 4286578816, 0], [0, 4278255615, 4278255615, 4278255615, 0], [0, 0, 0, 0, 0]], dtype='uint32')
    arr_numpy = tf.Image(data, dims=['x', 'y'])
    result_numpy = tf.shade(arr_numpy, color_key=color_key)
    assert (result_numpy.data == result).all()
    arr_dask = tf.Image(da.from_array(data, chunks=(2, 2)), dims=['x', 'y'])
    result_dask = tf.shade(arr_dask, color_key=color_key)
    assert (result_dask.data == result).all()
    try:
        import cupy
        arr_cupy = tf.Image(cupy.asarray(data), dims=['x', 'y'])
        result_cupy = tf.shade(arr_cupy, color_key=color_key)
        assert (result_cupy.data == result).all()
    except ImportError:
        cupy = None