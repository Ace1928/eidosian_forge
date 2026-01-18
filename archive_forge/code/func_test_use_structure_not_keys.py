from __future__ import annotations
import inspect
import pytest
import dask
from dask import delayed
from dask.base import collections_to_dsk, key_split, visualize_dsk
from dask.core import get_deps
from dask.order import _connecting_to_roots, diagnostics, ndependencies, order
from dask.utils_test import add, inc
def test_use_structure_not_keys(abcde):
    """See https://github.com/dask/dask/issues/5584#issuecomment-554963958

    We were using key names to infer structure, which could result in funny behavior.
    """
    a, b, _, _, _ = abcde
    dsk = {(a, 0): (f,), (a, 1): (f,), (a, 2): (f,), (a, 3): (f,), (a, 4): (f,), (a, 5): (f,), (a, 6): (f,), (a, 7): (f,), (a, 8): (f,), (a, 9): (f,), (b, 5): (f, (a, 2)), (b, 7): (f, (a, 0), (a, 2)), (b, 9): (f, (a, 7), (a, 0), (a, 2)), (b, 1): (f, (a, 4), (a, 7), (a, 0)), (b, 2): (f, (a, 9), (a, 4), (a, 7)), (b, 4): (f, (a, 6), (a, 9), (a, 4)), (b, 3): (f, (a, 5), (a, 6), (a, 9)), (b, 8): (f, (a, 1), (a, 5), (a, 6)), (b, 6): (f, (a, 8), (a, 1), (a, 5)), (b, 0): (f, (a, 3), (a, 8), (a, 1))}
    o = order(dsk)
    assert_topological_sort(dsk, o)
    assert max(diagnostics(dsk, o=o)[1]) == 3
    As = sorted((val for (letter, _), val in o.items() if letter == a))
    Bs = sorted((val for (letter, _), val in o.items() if letter == b))
    assert Bs[0] in {1, 3}
    if Bs[0] == 3:
        assert As == [0, 1, 2, 4, 6, 8, 10, 12, 14, 16]
        assert Bs == [3, 5, 7, 9, 11, 13, 15, 17, 18, 19]
    else:
        assert As == [0, 2, 4, 6, 8, 10, 12, 14, 16, 18]
        assert Bs == [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]