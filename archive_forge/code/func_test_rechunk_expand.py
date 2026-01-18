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
def test_rechunk_expand():
    a = np.random.default_rng().uniform(0, 1, 100).reshape((10, 10))
    x = da.from_array(a, chunks=(5, 5))
    y = x.rechunk(chunks=((3, 3, 3, 1), (3, 3, 3, 1)))
    assert np.all(y.compute() == a)