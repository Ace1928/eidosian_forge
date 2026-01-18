from __future__ import annotations
import numpy as np
import pytest
import dask.array as da
from dask import config
from dask.array.numpy_compat import AxisError
from dask.array.utils import assert_eq
def test_to_backend_cupy():
    with config.set({'array.backend': 'numpy'}):
        x = da.from_array(cupy.arange(11), chunks=(4,))
        assert isinstance(x._meta, cupy.ndarray)
        x_new = x.to_backend()
        assert isinstance(x_new._meta, np.ndarray)
        x_new = x.to_backend('cupy')
        assert isinstance(x_new._meta, cupy.ndarray)
        with config.set({'array.backend': 'cupy'}):
            x_new = x.to_backend('numpy')
            assert isinstance(x_new._meta, np.ndarray)
            x_new = x.to_backend()
            assert isinstance(x_new._meta, cupy.ndarray)
    assert_eq(x, x.to_backend('numpy'), check_type=False)
    assert_eq(x, x.to_backend('cupy'))