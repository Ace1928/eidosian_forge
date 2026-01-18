from __future__ import absolute_import, print_function, division
import pytest
from petl.errors import FieldSelectionError
from petl.test.helpers import ieq, eq_
from petl.comparison import Comparable
from petl.transform.selects import select, selectin, selectcontains, \
def test_select_empty():
    table = (('foo', 'bar'),)
    expect = (('foo', 'bar'),)
    actual = select(table, lambda r: r['foo'] == r['bar'])
    ieq(expect, actual)