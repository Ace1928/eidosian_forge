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
def test_slice_optimizations():
    expected = {('foo', 0): ('bar', 0)}
    result, chunks = slice_array('foo', 'bar', [[100]], (slice(None, None, None),), 8)
    assert expected == result
    expected = {('foo', 0): ('bar', 0), ('foo', 1): ('bar', 1), ('foo', 2): ('bar', 2)}
    result, chunks = slice_array('foo', 'bar', [(100, 1000, 10000)], (slice(None, None, None), slice(None, None, None), slice(None, None, None)), itemsize=8)
    assert expected == result