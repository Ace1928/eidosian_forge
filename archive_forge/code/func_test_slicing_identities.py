from __future__ import annotations
import itertools
import warnings
import pytest
from tlz import merge
import dask
import dask.array as da
from dask import config
from dask.array.chunk import getitem
from dask.array.slicing import (
from dask.array.utils import assert_eq, same_keys
def test_slicing_identities():
    a = da.ones((24, 16), chunks=((4, 8, 8, 4), (2, 6, 6, 2)))
    assert a is a[slice(None)]
    assert a is a[:]
    assert a is a[:]
    assert a is a[...]
    assert a is a[0:]
    assert a is a[0:]
    assert a is a[::1]
    assert a is a[0:len(a)]
    assert a is a[0::1]
    assert a is a[0:len(a):1]