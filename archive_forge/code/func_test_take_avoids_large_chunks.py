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
def test_take_avoids_large_chunks():
    with dask.config.set({'array.slicing.split-large-chunks': True}):
        chunks = ((1, 1, 1, 1), (500,), (500,))
        itemsize = 8
        index = np.array([0, 1] + [2] * 101 + [3])
        chunks2, dsk = take('a', 'b', chunks, index, itemsize)
        assert chunks2 == ((1, 1, 51, 50, 1), (500,), (500,))
        assert len(dsk) == 5
        index = np.array([0] * 101 + [1, 2, 3])
        chunks2, dsk = take('a', 'b', chunks, index, itemsize)
        assert chunks2 == ((51, 50, 1, 1, 1), (500,), (500,))
        assert len(dsk) == 5
        index = np.array([0, 1, 2] + [3] * 101)
        chunks2, dsk = take('a', 'b', chunks, index, itemsize)
        assert chunks2 == ((1, 1, 1, 51, 50), (500,), (500,))
        assert len(dsk) == 5
        chunks = ((500,), (1, 1, 1, 1), (500,))
        index = np.array([0, 1, 2] + [3] * 101)
        chunks2, dsk = take('a', 'b', chunks, index, itemsize, axis=1)
        assert chunks2 == ((500,), (1, 1, 1, 51, 50), (500,))
        assert len(dsk) == 5