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
def test_rechunk_0d():
    a = np.array(42)
    x = da.from_array(a, chunks=())
    y = x.rechunk(())
    assert y.chunks == ()
    assert y.compute() == a