from __future__ import annotations
import inspect
import pytest
import dask
from dask import delayed
from dask.base import collections_to_dsk, key_split, visualize_dsk
from dask.core import get_deps
from dask.order import _connecting_to_roots, diagnostics, ndependencies, order
from dask.utils_test import add, inc
def test_xarray_like_reduction():
    a, b, c, d, e = list('abcde')
    dsk = {}
    for ix in range(3):
        part = {(a, 0, ix): (f,), (a, 1, ix): (f,), (b, 0, ix): (f, (a, 0, ix)), (b, 1, ix): (f, (a, 0, ix), (a, 1, ix)), (b, 2, ix): (f, (a, 1, ix)), (c, 0, ix): (f, (b, 0, ix)), (c, 1, ix): (f, (b, 1, ix)), (c, 2, ix): (f, (b, 2, ix))}
        dsk.update(part)
    for ix in range(3):
        dsk.update({(d, ix): (f, (c, ix, 0), (c, ix, 1), (c, ix, 2))})
    o = order(dsk)
    assert_topological_sort(dsk, o)
    _, pressure = diagnostics(dsk, o=o)
    assert max(pressure) <= 9