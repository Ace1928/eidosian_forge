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
def test_swapaxes():
    x = np.random.default_rng().normal(0, 10, size=(10, 12, 7))
    d = da.from_array(x, chunks=(4, 5, 2))
    assert_eq(np.swapaxes(x, 0, 1), da.swapaxes(d, 0, 1))
    assert_eq(np.swapaxes(x, 2, 1), da.swapaxes(d, 2, 1))
    assert_eq(x.swapaxes(2, 1), d.swapaxes(2, 1))
    assert_eq(x.swapaxes(0, 0), d.swapaxes(0, 0))
    assert_eq(x.swapaxes(1, 2), d.swapaxes(1, 2))
    assert_eq(x.swapaxes(0, -1), d.swapaxes(0, -1))
    assert_eq(x.swapaxes(-1, 1), d.swapaxes(-1, 1))
    assert d.swapaxes(0, 1).name == d.swapaxes(0, 1).name
    assert d.swapaxes(0, 1).name != d.swapaxes(1, 0).name