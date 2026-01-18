from __future__ import absolute_import, print_function, division
from datetime import datetime
from petl.test.helpers import ieq
from petl.transform.setops import complement, intersection, diff, \
def test_recordcomplement_3():
    table1 = (('foo', 'bar'), ('A', 1), ('B', 2))
    table2 = (('bar', 'foo'),)
    expectation = (('foo', 'bar'), ('A', 1), ('B', 2))
    result = recordcomplement(table1, table2)
    ieq(expectation, result)
    ieq(expectation, result)
    expectation = (('bar', 'foo'),)
    result = recordcomplement(table2, table1)
    ieq(expectation, result)
    ieq(expectation, result)