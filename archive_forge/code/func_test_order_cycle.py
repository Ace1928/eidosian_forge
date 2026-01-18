from __future__ import annotations
import inspect
import pytest
import dask
from dask import delayed
from dask.base import collections_to_dsk, key_split, visualize_dsk
from dask.core import get_deps
from dask.order import _connecting_to_roots, diagnostics, ndependencies, order
from dask.utils_test import add, inc
def test_order_cycle():
    with pytest.raises(RuntimeError, match='Cycle detected'):
        dask.get({'a': (f, 'a')}, 'a')
    with pytest.raises(RuntimeError, match='Cycle detected'):
        order({'a': (f, 'a')})
    with pytest.raises(RuntimeError, match='Cycle detected'):
        order({('a', 0): (f, ('a', 0))})
    with pytest.raises(RuntimeError, match='Cycle detected'):
        order({'a': (f, 'b'), 'b': (f, 'c'), 'c': (f, 'a')})
    with pytest.raises(RuntimeError, match='Cycle detected'):
        order({'a': (f, 'b'), 'b': (f, 'c'), 'c': (f, 'a', 'd'), 'd': 1})
    with pytest.raises(RuntimeError, match='Cycle detected'):
        order({'a': (f, 'b'), 'b': (f, 'c'), 'c': (f, 'a', 'd'), 'd': (f, 'b')})