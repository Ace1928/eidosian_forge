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
def test_rechunk_minus_one():
    x = da.ones((24, 24), chunks=(4, 8))
    y = x.rechunk((-1, 8))
    assert y.chunks == ((24,), (8, 8, 8))
    assert_eq(x, y)