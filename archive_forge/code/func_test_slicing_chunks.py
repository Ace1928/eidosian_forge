from __future__ import annotations
import itertools
import warnings
import pytest
from tlz import merge
import dask
import dask.array as da
from dask import config
from dask.array.chunk import getitem
from dask.array.slicing import (
from dask.array.utils import assert_eq, same_keys
def test_slicing_chunks():
    result, chunks = slice_array('y', 'x', ([5, 5], [5, 5]), (1, np.array([2, 0, 3])), itemsize=8)
    assert chunks == ((3,),)
    result, chunks = slice_array('y', 'x', ([5, 5], [5, 5]), (slice(0, 7), np.array([2, 0, 3])), itemsize=8)
    assert chunks == ((5, 2), (3,))
    result, chunks = slice_array('y', 'x', ([5, 5], [5, 5]), (slice(0, 7), 1), itemsize=8)
    assert chunks == ((5, 2),)