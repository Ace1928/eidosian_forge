from __future__ import absolute_import, print_function, division
from datetime import datetime
from petl.test.helpers import ieq
from petl.transform.setops import complement, intersection, diff, \
def _test_intersection_2(intersection_impl):
    table1 = (('foo', 'bar', 'baz'), ('A', 1, True), ('C', 7, False), ('B', 2, False), ('C', 9, True))
    table2 = (('x', 'y', 'z'), ('B', 2, False), ('A', 9, False), ('B', 3, True), ('C', 9, True))
    expect = (('foo', 'bar', 'baz'), ('B', 2, False), ('C', 9, True))
    table3 = intersection_impl(table1, table2)
    ieq(expect, table3)