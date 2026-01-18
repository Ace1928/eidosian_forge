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
def test_slicing_consistent_names_after_normalization():
    x = da.zeros(10, chunks=(5,))
    assert same_keys(x[0:], x[:10])
    assert same_keys(x[0:], x[0:10])
    assert same_keys(x[0:], x[0:10:1])
    assert same_keys(x[:], x[0:10:1])