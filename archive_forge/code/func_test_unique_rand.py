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
@pytest.mark.parametrize('seed', [23, 796])
@pytest.mark.parametrize('shape, chunks', [[(10,), (5,)], [(10,), (3,)], [(4, 5), (3, 2)], [(20, 20), (4, 5)]])
def test_unique_rand(seed, shape, chunks):
    rng = np.random.default_rng(seed)
    a = rng.integers(0, 10, size=shape)
    d = da.from_array(a, chunks=chunks)
    r_a = np.unique(a, return_index=True, return_inverse=True, return_counts=True)
    r_d = da.unique(d, return_index=True, return_inverse=True, return_counts=True)
    assert_eq(r_d[0], r_a[0])
    assert_eq(r_d[1], r_a[1])
    assert_eq(r_d[2], r_a[2])
    assert_eq(r_d[3], r_a[3])