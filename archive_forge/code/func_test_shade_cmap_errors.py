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
def test_shade_cmap_errors(agg):
    with pytest.raises(ValueError):
        tf.shade(agg.a, cmap='foo')
    with pytest.raises(ValueError):
        tf.shade(agg.a, cmap=[])