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
def test_take_sorted():
    chunks, dsk = take('y', 'x', [(20, 20, 20, 20)], [1, 3, 5, 47], itemsize=8, axis=0)
    expected = {('y', 0): (getitem, ('x', 0), ([1, 3, 5],)), ('y', 1): (getitem, ('x', 2), ([7],))}
    np.testing.assert_equal(dsk, expected)
    assert chunks == ((3, 1),)
    chunks, dsk = take('y', 'x', [(20, 20, 20, 20), (20, 20)], [1, 3, 5, 37], itemsize=8, axis=1)
    expected = merge({('y', i, 0): (getitem, ('x', i, 0), (slice(None, None, None), [1, 3, 5])) for i in range(4)}, {('y', i, 1): (getitem, ('x', i, 1), (slice(None, None, None), [17])) for i in range(4)})
    np.testing.assert_equal(dsk, expected)
    assert chunks == ((20, 20, 20, 20), (3, 1))