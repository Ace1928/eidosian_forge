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
def test_where_nonzero():
    for shape, chunks in [(0, ()), ((0, 0), (0, 0)), ((15, 16), (4, 5))]:
        x = np.random.default_rng().integers(10, size=shape)
        d = da.from_array(x, chunks=chunks)
        x_w = np.where(x)
        d_w = da.where(d)
        assert isinstance(d_w, type(x_w))
        assert len(d_w) == len(x_w)
        for i in range(len(x_w)):
            assert_eq(d_w[i], x_w[i])