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
def test_piecewise_otherwise():
    rng = np.random.default_rng(1337)
    x = rng.integers(10, size=(15, 16))
    d = da.from_array(x, chunks=(4, 5))
    assert_eq(np.piecewise(x, [x > 5, x <= 2], [lambda e, v, k: e + 1, lambda e, v, k: v * e, lambda e, v, k: 0], 1, k=2), da.piecewise(d, [d > 5, d <= 2], [lambda e, v, k: e + 1, lambda e, v, k: v * e, lambda e, v, k: 0], 1, k=2))