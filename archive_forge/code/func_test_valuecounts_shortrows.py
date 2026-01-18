from __future__ import absolute_import, print_function, division
from petl.compat import PY2
from petl.test.helpers import ieq, eq_
from petl.util.counting import valuecount, valuecounter, valuecounts, \
def test_valuecounts_shortrows():
    table = (('foo', 'bar'), ('a', True), ('x', True), ('b',), ('b', True), ('c', False), ('z', False))
    actual = valuecounts(table, 'bar')
    expect = (('bar', 'count', 'frequency'), (True, 3, 3.0 / 6), (False, 2, 2.0 / 6), (None, 1, 1.0 / 6))
    ieq(expect, actual)
    ieq(expect, actual)