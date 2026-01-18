from __future__ import annotations
import contextlib
import itertools
import pickle
import sys
import warnings
from numbers import Number
import pytest
import dask
from dask.delayed import delayed
import dask.array as da
from dask.array.numpy_compat import NUMPY_GE_123, NUMPY_GE_200, AxisError
from dask.array.utils import assert_eq, same_keys
@pytest.mark.parametrize('shape', [(10,), (5, 10), (5, 10, 10)])
def test_shape_and_ndim(shape):
    x = da.random.default_rng().random(shape)
    assert np.shape(x) == shape
    x = da.random.default_rng().random(shape)
    assert np.ndim(x) == len(shape)