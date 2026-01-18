from __future__ import annotations
import warnings
from itertools import product
import pytest
import math
import dask
import dask.array as da
from dask.array.rechunk import (
from dask.array.utils import assert_eq
from dask.utils import funcname
def test_rechunk_blockshape():
    """Test that blockshape can be used."""
    new_shape, new_chunks = ((10, 10), (4, 3))
    new_blockdims = normalize_chunks(new_chunks, new_shape)
    old_chunks = ((4, 4, 2), (3, 3, 3, 1))
    a = np.random.default_rng().uniform(0, 1, 100).reshape((10, 10))
    x = da.from_array(a, chunks=old_chunks)
    check1 = rechunk(x, chunks=new_chunks)
    assert check1.chunks == new_blockdims
    assert np.all(check1.compute() == a)