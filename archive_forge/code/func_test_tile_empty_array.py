from __future__ import annotations
import pytest
import numpy as np
import pytest
from tlz import concat
import dask
import dask.array as da
from dask.array.core import normalize_chunks
from dask.array.numpy_compat import AxisError
from dask.array.utils import assert_eq, same_keys
@pytest.mark.parametrize('shape, chunks', [((1, 1, 0), (1, 1, 0)), ((2, 0), (1, 0))])
@pytest.mark.parametrize('reps', [2, (3, 2, 5)])
def test_tile_empty_array(shape, chunks, reps):
    x = np.empty(shape)
    d = da.from_array(x, chunks=chunks)
    assert_eq(np.tile(x, reps), da.tile(d, reps))