from __future__ import absolute_import, print_function, division
from datetime import datetime
from petl.test.helpers import ieq
from petl.transform.setops import complement, intersection, diff, \
def _test_complement_3(complement_impl):
    table1 = (('foo', 'bar'), ('A', 1), ('B', 2))
    table2 = (('foo', 'bar'),)
    expectation = (('foo', 'bar'), ('A', 1), ('B', 2))
    result = complement_impl(table1, table2)
    ieq(expectation, result)
    ieq(expectation, result)
    expectation = (('foo', 'bar'),)
    result = complement_impl(table2, table1)
    ieq(expectation, result)
    ieq(expectation, result)