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
def test_slicing_with_numpy_arrays():
    a, bd1 = slice_array('y', 'x', ((3, 3, 3, 1), (3, 3, 3, 1)), (np.array([1, 2, 9]), slice(None, None, None)), itemsize=8)
    b, bd2 = slice_array('y', 'x', ((3, 3, 3, 1), (3, 3, 3, 1)), (np.array([1, 2, 9]), slice(None, None, None)), itemsize=8)
    assert bd1 == bd2
    np.testing.assert_equal(a, b)
    i = [False, True, True, False, False, False, False, False, False, True]
    index = (i, slice(None, None, None))
    index = normalize_index(index, (10, 10))
    c, bd3 = slice_array('y', 'x', ((3, 3, 3, 1), (3, 3, 3, 1)), index, itemsize=8)
    assert bd1 == bd3
    np.testing.assert_equal(a, c)