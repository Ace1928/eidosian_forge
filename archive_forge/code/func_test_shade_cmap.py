from __future__ import annotations
from io import BytesIO
import numpy as np
import xarray as xr
import dask.array as da
import PIL
import pytest
import datashader.transfer_functions as tf
from datashader.tests.test_pandas import assert_eq_ndarray, assert_eq_xr, assert_image_close
@pytest.mark.parametrize('agg', aggs)
def test_shade_cmap(agg):
    cmap = ['red', (0, 255, 0), '#0000FF']
    img = tf.shade(agg.a, how='log', cmap=cmap)
    sol = np.array([[0, 4278190335, 4278236489], [4280344064, 0, 4289091584], [4292225024, 4294901760, 0]])
    sol = tf.Image(sol, coords=coords, dims=dims)
    assert_eq_xr(img, sol)