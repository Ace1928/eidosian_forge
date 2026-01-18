from __future__ import annotations
import inspect
import pytest
import dask
from dask import delayed
from dask.base import collections_to_dsk, key_split, visualize_dsk
from dask.core import get_deps
from dask.order import _connecting_to_roots, diagnostics, ndependencies, order
from dask.utils_test import add, inc
def test_prefer_short_ancestor(abcde):
    """
    From https://github.com/dask/dask-ml/issues/206#issuecomment-395869929

    Two cases, one where chunks of an array are independent, and one where the
    chunks of an array have a shared source. We handled the independent one
    "well" earlier.

    Good:

                    c2
                   / \\ \\
                  /   \\ \\
                c1     \\ \\
              / | \\     \\ \\
            c0  a0 b0   a1 b1

    Bad:

                    c2
                   / \\ \\
                  /   \\ \\
                c1     \\ \\
              / | \\     \\ \\
            c0  a0 b0   a1 b1
                   \\ \\   / /
                    \\ \\ / /
                      a-b


    The difference is that all the `a` and `b` tasks now have a common
    ancestor.

    We would like to choose c1 *before* a1, and b1 because

    * we can release a0 and b0 once c1 is done
    * we don't need a1 and b1 to compute c1.
    """
    a, b, c, _, _ = abcde
    ab = a + b
    dsk = {ab: 0, (a, 0): (f, ab, 0, 0), (b, 0): (f, ab, 0, 1), (c, 0): 0, (c, 1): (f, (c, 0), (a, 0), (b, 0)), (a, 1): (f, ab, 1, 0), (b, 1): (f, ab, 1, 1), (c, 2): (f, (c, 1), (a, 1), (b, 1))}
    o = order(dsk)
    assert_topological_sort(dsk, o)
    assert o[a, 0] < o[a, 1]
    assert o[b, 0] < o[b, 1]
    assert o[b, 0] < o[c, 2]
    assert o[c, 1] < o[c, 2]
    assert o[c, 1] < o[a, 1]