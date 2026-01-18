from __future__ import absolute_import, print_function, division
from petl.compat import PY2
from petl.test.helpers import ieq, eq_
from petl.util.counting import valuecount, valuecounter, valuecounts, \
def test_parsecounts():
    table = (('foo', 'bar', 'baz'), ('A', 'aaa', 2), ('B', '2', '3.4'), ('B', '3', '7.8', True), ('D', '3.7', 9.0), ('E', 42))
    actual = parsecounts(table, 'bar')
    expect = (('type', 'count', 'errors'), ('float', 3, 1), ('int', 2, 2))
    ieq(expect, actual)