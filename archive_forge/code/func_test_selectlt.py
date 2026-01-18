from __future__ import absolute_import, print_function, division
import pytest
from petl.errors import FieldSelectionError
from petl.test.helpers import ieq, eq_
from petl.comparison import Comparable
from petl.transform.selects import select, selectin, selectcontains, \
def test_selectlt():
    table = (('foo', 'bar', 'baz'), ('a', 4, 9.3), ('a', 2, 88.2), ('b', 1, None), ('c', 8, 42.0), ('d', 7, 100.9), ('c', 2))
    actual = selectlt(table, 'baz', 50)
    expect = (('foo', 'bar', 'baz'), ('a', 4, 9.3), ('b', 1, None), ('c', 8, 42.0), ('c', 2))
    ieq(expect, actual)
    ieq(expect, actual)