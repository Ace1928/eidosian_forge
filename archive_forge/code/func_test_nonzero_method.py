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
def test_nonzero_method():
    for shape, chunks in [(0, ()), ((0, 0), (0, 0)), ((15, 16), (4, 5))]:
        x = np.random.default_rng().integers(10, size=shape)
        d = da.from_array(x, chunks=chunks)
        x_nz = x.nonzero()
        d_nz = d.nonzero()
        assert isinstance(d_nz, type(x_nz))
        assert len(d_nz) == len(x_nz)
        for i in range(len(x_nz)):
            assert_eq(d_nz[i], x_nz[i])