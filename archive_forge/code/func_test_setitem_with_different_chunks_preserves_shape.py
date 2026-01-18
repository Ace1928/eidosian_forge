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
@pytest.mark.parametrize('params', [(2, 2, 1), (5, 3, 2)])
def test_setitem_with_different_chunks_preserves_shape(params):
    """Reproducer for https://github.com/dask/dask/issues/3730.

    Mutating based on an array with different chunks can cause new chunks to be
    used.  We need to ensure those new chunk sizes are applied to the mutated
    array, otherwise the array won't generate the correct keys.
    """
    array_size, chunk_size1, chunk_size2 = params
    x = da.zeros(array_size, chunks=chunk_size1)
    mask = da.zeros(array_size, chunks=chunk_size2)
    x[mask] = 1
    result = x.compute()
    assert x.shape == result.shape