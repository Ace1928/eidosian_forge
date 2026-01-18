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
def test_rechunk_avoid_needless_chunking():
    x = da.ones(16, chunks=2)
    y = x.rechunk(8)
    dsk = y.__dask_graph__()
    assert len(dsk) <= 8 + 2