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
def test_rechunk_zero():
    with dask.config.set({'array.chunk-size': '1B'}):
        x = da.ones(10, chunks=(5,))
        y = x.rechunk('auto')
        assert y.chunks == ((1,) * 10,)