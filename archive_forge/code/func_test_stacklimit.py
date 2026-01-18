from __future__ import annotations
import inspect
import pytest
import dask
from dask import delayed
from dask.base import collections_to_dsk, key_split, visualize_dsk
from dask.core import get_deps
from dask.order import _connecting_to_roots, diagnostics, ndependencies, order
from dask.utils_test import add, inc
def test_stacklimit(abcde):
    dsk = {'x%s' % (i + 1): (inc, 'x%s' % i) for i in range(10000)}
    dependencies, dependents = get_deps(dsk)
    ndependencies(dependencies, dependents)