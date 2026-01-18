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
def test_where_incorrect_args():
    a = da.ones(5, chunks=3)
    for kwd in ['x', 'y']:
        kwargs = {kwd: a}
        try:
            da.where(a > 0, **kwargs)
        except ValueError as e:
            assert 'either both or neither of x and y should be given' in str(e)