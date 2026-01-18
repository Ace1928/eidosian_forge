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
def test_rechunk_method():
    """Test rechunking can be done as a method of dask array."""
    old = ((5, 2, 3),) * 4
    new = ((3, 3, 3, 1),) * 4
    a = np.random.default_rng().uniform(0, 1, 10000).reshape((10,) * 4)
    x = da.from_array(a, chunks=old)
    x2 = x.rechunk(chunks=new)
    assert x2.chunks == new
    assert np.all(x2.compute() == a)