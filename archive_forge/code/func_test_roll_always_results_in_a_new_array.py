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
def test_roll_always_results_in_a_new_array():
    x = da.arange(2, 3)
    y = da.roll(x, 1)
    assert y is not x