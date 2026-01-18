from __future__ import annotations
from io import BytesIO
import numpy as np
import xarray as xr
import dask.array as da
import PIL
import pytest
import datashader.transfer_functions as tf
from datashader.tests.test_pandas import assert_eq_ndarray, assert_eq_xr, assert_image_close
def test_rgb_density():
    b = 4294901760
    data = np.full((4, 4), b, dtype='uint32')
    assert tf._rgb_density(data) == 1.0
    data = np.zeros((4, 4), dtype='uint32')
    assert tf._rgb_density(data) == np.inf
    data[3, 3] = b
    assert tf._rgb_density(data) == 0
    data[2, 0] = data[0, 2] = data[1, 1] = b
    assert np.allclose(tf._rgb_density(data), 0.75)
    assert np.allclose(tf._rgb_density(data, 3), 1)