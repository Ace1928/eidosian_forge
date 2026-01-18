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
def test_rechunk_same():
    x = da.ones((24, 24), chunks=(4, 8))
    y = x.rechunk(x.chunks)
    assert x is y