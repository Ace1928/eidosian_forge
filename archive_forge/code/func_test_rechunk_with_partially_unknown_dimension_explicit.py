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
@pytest.mark.parametrize('new_chunks', [((5, 5, np.nan, np.nan), (5, 5)), ((5, 5, math.nan, math.nan), (5, 5)), ((5, 5, float('nan'), float('nan')), (5, 5))])
def test_rechunk_with_partially_unknown_dimension_explicit(new_chunks):
    dd = pytest.importorskip('dask.dataframe')
    x = da.ones(shape=(10, 10), chunks=(5, 2))
    y = dd.from_array(x).values
    z = da.concatenate([x, y])
    xx = da.concatenate([x, x])
    result = z.rechunk(new_chunks)
    expected = xx.rechunk((None, (5, 5)))
    assert_chunks_match(result.chunks, expected.chunks)
    assert_eq(result, expected)