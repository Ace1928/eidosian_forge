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
@pytest.mark.parametrize('shape', [(1,), (1, 1)])
def test_squeeze_1d_array(shape):
    a = np.full(shape=shape, fill_value=2)
    a_s = np.squeeze(a)
    d = da.from_array(a, chunks=1)
    d_s = da.squeeze(d)
    assert isinstance(d_s, da.Array)
    assert isinstance(d_s.compute(), np.ndarray)
    assert_eq(d_s, a_s)