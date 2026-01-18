from __future__ import annotations
import inspect
import pytest
import dask
from dask import delayed
from dask.base import collections_to_dsk, key_split, visualize_dsk
from dask.core import get_deps
from dask.order import _connecting_to_roots, diagnostics, ndependencies, order
from dask.utils_test import add, inc
def test_type_comparisions_ok(abcde):
    a, b, c, d, e = abcde
    dsk = {a: 1, (a, 1): 2, (a, b, 1): 3}
    order(dsk)