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
@pytest.mark.parametrize('n, k, m, chunks', [(3, 0, 3, 'auto'), (3, 1, 3, 'auto'), (3, -1, 3, 'auto'), (5, 0, 5, 1)])
def test_tril_triu_indices(n, k, m, chunks):
    actual = da.tril_indices(n=n, k=k, m=m, chunks=chunks)[0]
    expected = np.tril_indices(n=n, k=k, m=m)[0]
    if sys.platform == 'win32':
        assert_eq(actual.astype(expected.dtype), expected)
    else:
        assert_eq(actual, expected)
    actual = da.triu_indices(n=n, k=k, m=m, chunks=chunks)[0]
    expected = np.triu_indices(n=n, k=k, m=m)[0]
    if sys.platform == 'win32':
        assert_eq(actual.astype(expected.dtype), expected)
    else:
        assert_eq(actual, expected)