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
def test_where_bool_optimization():
    rng = np.random.default_rng()
    x = rng.integers(10, size=(15, 16))
    d = da.from_array(x, chunks=(4, 5))
    y = rng.integers(10, size=(15, 16))
    e = da.from_array(y, chunks=(4, 5))
    for c in [True, False, np.True_, np.False_, 1, 0]:
        w1 = da.where(c, d, e)
        w2 = np.where(c, x, y)
        assert_eq(w1, w2)
        ex_w1 = d if c else e
        assert w1 is ex_w1