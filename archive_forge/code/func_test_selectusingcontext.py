from __future__ import absolute_import, print_function, division
import pytest
from petl.errors import FieldSelectionError
from petl.test.helpers import ieq, eq_
from petl.comparison import Comparable
from petl.transform.selects import select, selectin, selectcontains, \
def test_selectusingcontext():
    table1 = (('foo', 'bar'), ('A', 1), ('B', 4), ('C', 5), ('D', 9))
    expect = (('foo', 'bar'), ('B', 4), ('C', 5))

    def query(prv, cur, nxt):
        return prv is not None and cur.bar - prv.bar < 2 or (nxt is not None and nxt.bar - cur.bar < 2)
    actual = selectusingcontext(table1, query)
    ieq(expect, actual)
    ieq(expect, actual)