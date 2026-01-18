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
def test_rechunk_auto_3d():
    x = da.ones((20, 20, 20), chunks=(2, 2, 2))
    y = x.rechunk({0: 'auto', 1: 'auto'}, block_size_limit=200 * x.dtype.itemsize)
    assert y.chunks[2] == x.chunks[2]
    assert y.chunks[0] == (10, 10)
    assert y.chunks[1] == (10, 10)