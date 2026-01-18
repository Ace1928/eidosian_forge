from __future__ import annotations
import inspect
import pytest
import dask
from dask import delayed
from dask.base import collections_to_dsk, key_split, visualize_dsk
from dask.core import get_deps
from dask.order import _connecting_to_roots, diagnostics, ndependencies, order
from dask.utils_test import add, inc
def test_recursion_depth_long_linear_chains():
    dsk = {'-1': (f, 1)}
    for ix in range(10000):
        dsk[str(ix)] = (f, str(ix - 1))
    assert len(order(dsk)) == len(dsk)