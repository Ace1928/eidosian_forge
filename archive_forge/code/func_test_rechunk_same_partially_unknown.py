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
def test_rechunk_same_partially_unknown():
    dd = pytest.importorskip('dask.dataframe')
    x = da.ones(shape=(10, 10), chunks=(5, 10))
    y = dd.from_array(x).values
    z = da.concatenate([x, y])
    new_chunks = ((5, 5, np.nan, np.nan), (10,))
    assert z.chunks == new_chunks
    result = z.rechunk(new_chunks)
    assert z is result