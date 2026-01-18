from __future__ import annotations
import pytest
import numpy as np
import dask
import dask.array as da
from dask.array.chunk import getitem as da_getitem
from dask.array.core import getter as da_getter
from dask.array.core import getter_nofancy as da_getter_nofancy
from dask.array.optimization import (
from dask.array.utils import assert_eq
from dask.highlevelgraph import HighLevelGraph
from dask.optimization import SubgraphCallable, fuse
from dask.utils import SerializableLock
def test_optimize_slicing(getter):
    dsk = {'a': (range, 10), 'b': (getter, 'a', (slice(None, None, None),)), 'c': (getter, 'b', (slice(None, None, None),)), 'd': (getter, 'c', (slice(0, 5, None),)), 'e': (getter, 'd', (slice(None, None, None),))}
    expected = {'e': (getter, (range, 10), (slice(0, 5, None),))}
    result = optimize_slices(fuse(dsk, [], rename_keys=False)[0])
    _assert_getter_dsk_eq(result, expected)
    expected = {'c': (getter, (range, 10), (slice(0, None, None),)), 'd': (getter, 'c', (slice(0, 5, None),)), 'e': (getter, 'd', (slice(None, None, None),))}
    result = optimize_slices(fuse(dsk, ['c', 'd', 'e'], rename_keys=False)[0])
    _assert_getter_dsk_eq(result, expected)