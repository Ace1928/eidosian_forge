from __future__ import annotations
import inspect
import pytest
import dask
from dask import delayed
from dask.base import collections_to_dsk, key_split, visualize_dsk
from dask.core import get_deps
from dask.order import _connecting_to_roots, diagnostics, ndependencies, order
from dask.utils_test import add, inc
def test_run_smaller_sections(abcde):
    """
            aa
           / |
      b   d  bb dd
     / \\ /|  | /
    a   c e  cc

    """
    a, b, c, d, e = abcde
    aa, bb, cc, dd = (x * 2 for x in [a, b, c, d])
    dsk = {a: (f,), c: (f,), e: (f,), cc: (f,), b: (f, a, c), d: (f, c, e), bb: (f, cc), aa: (f, d, bb), dd: (f, cc)}
    o = order(dsk)
    assert_topological_sort(dsk, o)
    assert max(diagnostics(dsk)[1]) <= 4
    assert o[aa] < o[a] and o[dd] < o[a] or (o[b] < o[e] and o[b] < o[cc])