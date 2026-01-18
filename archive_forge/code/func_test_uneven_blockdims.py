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
def test_uneven_blockdims():
    blockdims = ((31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30), (100,))
    index = (slice(240, 270), slice(None))
    dsk_out, bd_out = slice_array('in', 'out', blockdims, index, itemsize=8)
    sol = {('in', 0, 0): (getitem, ('out', 7, 0), (slice(28, 31, 1), slice(None))), ('in', 1, 0): (getitem, ('out', 8, 0), (slice(0, 27, 1), slice(None)))}
    assert dsk_out == sol
    assert bd_out == ((3, 27), (100,))
    blockdims = ((31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30),) * 2
    index = (slice(240, 270), slice(180, 230))
    dsk_out, bd_out = slice_array('in', 'out', blockdims, index, itemsize=8)
    sol = {('in', 0, 0): (getitem, ('out', 7, 5), (slice(28, 31, 1), slice(29, 30, 1))), ('in', 0, 1): (getitem, ('out', 7, 6), (slice(28, 31, 1), slice(None))), ('in', 0, 2): (getitem, ('out', 7, 7), (slice(28, 31, 1), slice(0, 18, 1))), ('in', 1, 0): (getitem, ('out', 8, 5), (slice(0, 27, 1), slice(29, 30, 1))), ('in', 1, 1): (getitem, ('out', 8, 6), (slice(0, 27, 1), slice(None))), ('in', 1, 2): (getitem, ('out', 8, 7), (slice(0, 27, 1), slice(0, 18, 1)))}
    assert dsk_out == sol
    assert bd_out == ((3, 27), (1, 31, 18))