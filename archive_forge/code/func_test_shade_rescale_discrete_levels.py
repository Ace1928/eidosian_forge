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
@pytest.mark.parametrize('attr', ['d'])
@pytest.mark.parametrize('rescale', [False, True])
def test_shade_rescale_discrete_levels(agg, attr, rescale):
    x = getattr(agg, attr)
    cmap = ['pink', 'red']
    img = tf.shade(x, cmap=cmap, how='eq_hist', rescale_discrete_levels=rescale)
    if rescale:
        sol = np.array([[4287201791, 4285360127, 4283583999], [4281742335, 4287201791, 4279966207], [4278190335, 4287201791, 4287201791]], dtype='uint32')
    else:
        sol = np.array([[4291543295, 4288846335, 4286149631], [4283518207, 4291543295, 4280821503], [4278190335, 4291543295, 4291543295]], dtype='uint32')
    sol = tf.Image(sol, coords=coords, dims=dims)
    assert_eq_xr(img, sol)