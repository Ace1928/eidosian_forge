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
@pytest.mark.filterwarnings('ignore:Slicing:dask.array.core.PerformanceWarning')
@pytest.mark.parametrize('size, chunks', [((100, 2), (50, 2)), ((100, 2), (37, 1)), ((100,), (55,))])
def test_shuffle_slice(size, chunks):
    x = da.random.default_rng().integers(0, 1000, size=size, chunks=chunks)
    index = np.arange(len(x))
    np.random.default_rng().shuffle(index)
    a = x[index]
    b = shuffle_slice(x, index)
    assert_eq(a, b)