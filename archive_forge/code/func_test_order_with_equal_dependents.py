from __future__ import annotations
import inspect
import pytest
import dask
from dask import delayed
from dask.base import collections_to_dsk, key_split, visualize_dsk
from dask.core import get_deps
from dask.order import _connecting_to_roots, diagnostics, ndependencies, order
from dask.utils_test import add, inc
def test_order_with_equal_dependents(abcde):
    """From https://github.com/dask/dask/issues/5859#issuecomment-608422198

    See the visualization of `(maxima, argmax)` example from the above comment.

    This DAG has enough structure to exercise more parts of `order`

    """
    max_pressure = 10
    a, b, c, d, e = abcde
    dsk = {}
    abc = [a, b, c, d]
    for x in abc:
        dsk.update({(x, 0): (f, 0), (x, 1): (f, (x, 0)), (x, 2, 0): (f, (x, 0)), (x, 2, 1): (f, (x, 1))})
        for i, y in enumerate(abc):
            dsk.update({(x, 3, i): (f, (x, 2, 0), (y, 2, 1)), (x, 4, i): (f, (x, 3, i)), (x, 5, i, 0): (f, (x, 4, i)), (x, 5, i, 1): (f, (x, 4, i)), (x, 6, i, 0): (f, (x, 5, i, 0)), (x, 6, i, 1): (f, (x, 5, i, 1))})
    o = order(dsk)
    assert_topological_sort(dsk, o)
    total = 0
    for x in abc:
        for i in range(len(abc)):
            val = abs(o[x, 6, i, 1] - o[x, 6, i, 0])
            total += val
    assert total <= 32
    pressure = diagnostics(dsk, o=o)[1]
    assert max(pressure) <= max_pressure
    dsk2 = dict(dsk)
    for x in abc:
        for i in range(len(abc)):
            dsk2[x, 7, i, 0] = (f, (x, 6, i, 0))
    o = order(dsk2)
    assert_topological_sort(dsk2, o)
    total = 0
    for x in abc:
        for i in range(len(abc)):
            val = abs(o[x, 6, i, 1] - o[x, 7, i, 0])
            total += val
    assert total <= 48
    pressure = diagnostics(dsk2, o=o)[1]
    assert max(pressure) <= max_pressure
    dsk3 = dict(dsk)
    for x in abc:
        for i in range(len(abc)):
            del dsk3[x, 6, i, 1]
    o = order(dsk3)
    assert_topological_sort(dsk3, o)
    total = 0
    for x in abc:
        for i in range(len(abc)):
            val = abs(o[x, 5, i, 1] - o[x, 6, i, 0])
            total += val
    assert total <= 32
    pressure = diagnostics(dsk3, o=o)[1]
    assert max(pressure) <= max_pressure
    dsk4 = dict(dsk3)
    for x in abc:
        for i in range(len(abc)):
            del dsk4[x, 6, i, 0]
    o = order(dsk4)
    assert_topological_sort(dsk4, o)
    pressure = diagnostics(dsk4, o=o)[1]
    assert max(pressure) <= max_pressure
    for x in abc:
        for i in range(len(abc)):
            assert abs(o[x, 5, i, 1] - o[x, 5, i, 0]) <= 2