from __future__ import absolute_import, print_function, division
import pytest
from petl.errors import FieldSelectionError
from petl.test.helpers import ieq, eq_
from petl.comparison import Comparable
from petl.transform.selects import select, selectin, selectcontains, \
def test_recordselect():
    table = (('foo', 'bar', 'baz'), ('a', 4, 9.3), ('a', 2, 88.2), ('b', 1, 23.3), ('c', 8, 42.0), ('d', 7, 100.9), ('c', 2))
    actual = select(table, lambda rec: rec['foo'] == 'a')
    expect = (('foo', 'bar', 'baz'), ('a', 4, 9.3), ('a', 2, 88.2))
    ieq(expect, actual)
    ieq(expect, actual)