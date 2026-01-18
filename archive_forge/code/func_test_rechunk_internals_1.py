from __future__ import annotations
import warnings
from itertools import product
import pytest
import math
import dask
import dask.array as da
from dask.array.rechunk import (
from dask.array.utils import assert_eq
from dask.utils import funcname
def test_rechunk_internals_1():
    """Test the cumdims_label and _breakpoints and
    _intersect_1d internal funcs to rechunk."""
    new = cumdims_label(((1, 1, 2), (1, 5, 1)), 'n')
    old = cumdims_label(((4,), (1,) * 5), 'o')
    breaks = tuple((_breakpoints(o, n) for o, n in zip(old, new)))
    answer = (('o', 0), ('n', 0), ('n', 1), ('n', 2), ('o', 4), ('n', 4))
    assert breaks[0] == answer
    answer2 = (('o', 0), ('n', 0), ('o', 1), ('n', 1), ('o', 2), ('o', 3), ('o', 4), ('o', 5), ('n', 6), ('n', 7))
    assert breaks[1] == answer2
    i1d = [_intersect_1d(b) for b in breaks]
    answer3 = [[(0, slice(0, 1))], [(0, slice(1, 2))], [(0, slice(2, 4))]]
    assert i1d[0] == answer3
    answer4 = [[(0, slice(0, 1))], [(1, slice(0, 1)), (2, slice(0, 1)), (3, slice(0, 1)), (4, slice(0, 1)), (5, slice(0, 1))], [(5, slice(1, 2))]]
    assert i1d[1] == answer4
    new = cumdims_label(((1, 1, 2), (1, 5, 1, 0)), 'n')
    breaks = tuple((_breakpoints(o, n) for o, n in zip(old, new)))
    answer5 = (('o', 0), ('n', 0), ('o', 1), ('n', 1), ('o', 2), ('o', 3), ('o', 4), ('o', 5), ('n', 6), ('n', 7), ('n', 7))
    assert breaks[1] == answer5
    i1d = [_intersect_1d(b) for b in breaks]
    answer6 = [[(0, slice(0, 1))], [(1, slice(0, 1)), (2, slice(0, 1)), (3, slice(0, 1)), (4, slice(0, 1)), (5, slice(0, 1))], [(5, slice(1, 2))], [(5, slice(2, 2))]]
    assert i1d[1] == answer6