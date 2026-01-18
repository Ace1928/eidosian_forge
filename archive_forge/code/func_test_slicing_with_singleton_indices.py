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
def test_slicing_with_singleton_indices():
    result, chunks = slice_array('y', 'x', ([5, 5], [5, 5]), (slice(0, 5), 8), itemsize=8)
    expected = {('y', 0): (getitem, ('x', 0, 1), (slice(None, None, None), 3))}
    assert expected == result