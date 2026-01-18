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
def test_tril_ndims():
    A = np.random.default_rng().integers(0, 11, (10, 10, 10))
    dA = da.from_array(A, chunks=(5, 5, 5))
    assert_eq(da.triu(dA), np.triu(A))