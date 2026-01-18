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
def test_rechunk_down():
    with dask.config.set({'array.chunk-size': '10MiB'}):
        x = da.ones((100, 1000, 1000), chunks=(1, 1000, 1000), dtype='uint8')
        y = x.rechunk('auto')
        assert y.chunks == ((10,) * 10, (1000,), (1000,))
    with dask.config.set({'array.chunk-size': '1MiB'}):
        z = y.rechunk('auto')
        assert z.chunks == ((4,) * 25, (511, 489), (511, 489))
    with dask.config.set({'array.chunk-size': '1MiB'}):
        z = y.rechunk({0: 'auto'})
        assert z.chunks == ((1,) * 100, (1000,), (1000,))
        z = y.rechunk({1: 'auto'})
        assert z.chunks == ((10,) * 10, (104,) * 9 + (64,), (1000,))