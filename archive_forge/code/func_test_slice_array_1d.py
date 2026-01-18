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
def test_slice_array_1d():
    expected = {('y', 0): (getitem, ('x', 0), (slice(24, 25, 2),)), ('y', 1): (getitem, ('x', 1), (slice(1, 25, 2),)), ('y', 2): (getitem, ('x', 2), (slice(0, 25, 2),)), ('y', 3): (getitem, ('x', 3), (slice(1, 25, 2),))}
    result, chunks = slice_array('y', 'x', [[25] * 4], [slice(24, None, 2)], 8)
    assert expected == result
    expected = {('y', 0): (getitem, ('x', 1), (slice(1, 25, 2),)), ('y', 1): (getitem, ('x', 2), (slice(0, 25, 2),)), ('y', 2): (getitem, ('x', 3), (slice(1, 25, 2),))}
    result, chunks = slice_array('y', 'x', [[25] * 4], [slice(26, None, 2)], 8)
    assert expected == result
    expected = {('y', 0): (getitem, ('x', 0), (slice(24, 25, 2),)), ('y', 1): (getitem, ('x', 1), (slice(1, 25, 2),)), ('y', 2): (getitem, ('x', 2), (slice(0, 25, 2),)), ('y', 3): (getitem, ('x', 3), (slice(1, 25, 2),))}
    result, chunks = slice_array('y', 'x', [(25,) * 4], (slice(24, None, 2),), 8)
    assert expected == result
    expected = {('y', 0): (getitem, ('x', 1), (slice(1, 25, 2),)), ('y', 1): (getitem, ('x', 2), (slice(0, 25, 2),)), ('y', 2): (getitem, ('x', 3), (slice(1, 25, 2),))}
    result, chunks = slice_array('y', 'x', [(25,) * 4], (slice(26, None, 2),), 8)
    assert expected == result