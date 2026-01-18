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
def test_select_return_dtype():
    d = np.array([1, 2, 3, np.nan, 5, 7])
    m = np.isnan(d)
    d_d = da.from_array(d)
    d_m = da.isnan(d_d)
    assert_eq(np.select([m], [d]), da.select([d_m], [d_d]), equal_nan=True)