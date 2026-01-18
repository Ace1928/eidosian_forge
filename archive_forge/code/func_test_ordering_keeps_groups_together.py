from __future__ import annotations
import inspect
import pytest
import dask
from dask import delayed
from dask.base import collections_to_dsk, key_split, visualize_dsk
from dask.core import get_deps
from dask.order import _connecting_to_roots, diagnostics, ndependencies, order
from dask.utils_test import add, inc
def test_ordering_keeps_groups_together(abcde):
    a, b, c, d, e = abcde
    d = {(a, i): (f,) for i in range(4)}
    d.update({(b, 0): (f, (a, 0), (a, 1)), (b, 1): (f, (a, 2), (a, 3))})
    o = order(d)
    assert_topological_sort(d, o)
    assert abs(o[a, 0] - o[a, 1]) == 1
    assert abs(o[a, 2] - o[a, 3]) == 1
    d = {(a, i): (f,) for i in range(4)}
    d.update({(b, 0): (f, (a, 0), (a, 2)), (b, 1): (f, (a, 1), (a, 3))})
    o = order(d)
    assert_topological_sort(d, o)
    assert abs(o[a, 0] - o[a, 2]) == 1
    assert abs(o[a, 1] - o[a, 3]) == 1