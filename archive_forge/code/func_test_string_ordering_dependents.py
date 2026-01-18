from __future__ import annotations
import inspect
import pytest
import dask
from dask import delayed
from dask.base import collections_to_dsk, key_split, visualize_dsk
from dask.core import get_deps
from dask.order import _connecting_to_roots, diagnostics, ndependencies, order
from dask.utils_test import add, inc
@pytest.mark.xfail(reason='see comment', strict=False)
def test_string_ordering_dependents():
    """Prefer ordering tasks by name first even when in dependencies"""
    dsk = {('a', 1): (f, 'b'), ('a', 2): (f, 'b'), ('a', 3): (f, 'b'), 'b': (f,)}
    o = order(dsk)
    assert_topological_sort(dsk, o)
    assert o == {'b': 0, ('a', 1): 1, ('a', 2): 2, ('a', 3): 3} or o == {'b': 0, ('a', 1): 3, ('a', 2): 2, ('a', 3): 1}