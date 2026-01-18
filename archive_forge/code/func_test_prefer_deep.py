from __future__ import annotations
import inspect
import pytest
import dask
from dask import delayed
from dask.base import collections_to_dsk, key_split, visualize_dsk
from dask.core import get_deps
from dask.order import _connecting_to_roots, diagnostics, ndependencies, order
from dask.utils_test import add, inc
def test_prefer_deep(abcde):
    """
        c
        |
    e   b
    |   |
    d   a

    Prefer longer chains first so we should start with c
    """
    a, b, c, d, e = abcde
    dsk = {a: 1, b: (f, a), c: (f, b), d: 1, e: (f, d)}
    o = order(dsk)
    assert_topological_sort(dsk, o)
    assert o[a] < o[d]
    assert o[b] < o[d]