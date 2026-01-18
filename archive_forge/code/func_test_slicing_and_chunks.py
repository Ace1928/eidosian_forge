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
def test_slicing_and_chunks():
    o = da.ones((24, 16), chunks=((4, 8, 8, 4), (2, 6, 6, 2)))
    t = o[4:-4, 2:-2]
    assert t.chunks == ((8, 8), (6, 6))