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
@pytest.mark.parametrize('shape, chunks', [((10,), (1,)), ((10, 11, 13), (4, 5, 3))])
@pytest.mark.parametrize('reps', [0, (0,), (2, 0), (0, 3, 0, 4)])
def test_tile_zero_reps(shape, chunks, reps):
    x = np.random.random(shape)
    d = da.from_array(x, chunks=chunks)
    assert_eq(np.tile(x, reps), da.tile(d, reps))