from __future__ import annotations
import inspect
import pytest
import dask
from dask import delayed
from dask.base import collections_to_dsk, key_split, visualize_dsk
from dask.core import get_deps
from dask.order import _connecting_to_roots, diagnostics, ndependencies, order
from dask.utils_test import add, inc
def test_order_doesnt_fail_on_mixed_type_keys(abcde):
    order({'x': (inc, 1), ('y', 0): (inc, 2), 'z': (add, 'x', ('y', 0))})