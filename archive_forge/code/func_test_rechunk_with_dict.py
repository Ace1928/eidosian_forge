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
def test_rechunk_with_dict():
    x = da.ones((24, 24), chunks=(4, 8))
    y = x.rechunk(chunks={0: 12})
    assert y.chunks == ((12, 12), (8, 8, 8))
    x = da.ones((24, 24), chunks=(4, 8))
    y = x.rechunk(chunks={0: (12, 12)})
    assert y.chunks == ((12, 12), (8, 8, 8))
    x = da.ones((24, 24), chunks=(4, 8))
    y = x.rechunk(chunks={0: -1})
    assert y.chunks == ((24,), (8, 8, 8))
    x = da.ones((24, 24), chunks=(4, 8))
    y = x.rechunk(chunks={0: None, 1: 'auto'})
    assert y.chunks == ((4, 4, 4, 4, 4, 4), (24,))