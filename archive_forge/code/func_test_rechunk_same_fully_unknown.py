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
def test_rechunk_same_fully_unknown():
    dd = pytest.importorskip('dask.dataframe')
    x = da.ones(shape=(10, 10), chunks=(5, 10))
    y = dd.from_array(x).values
    new_chunks = ((np.nan, np.nan), (10,))
    assert y.chunks == new_chunks
    result = y.rechunk(new_chunks)
    assert y is result