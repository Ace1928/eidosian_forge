from __future__ import absolute_import, print_function, division
from petl.compat import PY2
from petl.test.helpers import ieq, eq_
from petl.util.counting import valuecount, valuecounter, valuecounts, \
def test_rowlengths():
    table = (('foo', 'bar', 'baz'), ('A', 1, 2), ('B', '2', '3.4'), ('B', '3', '7.8', True), ('D', 'xyz', 9.0), ('E', None), ('F', 9))
    actual = rowlengths(table)
    expect = (('length', 'count'), (3, 3), (2, 2), (4, 1))
    ieq(expect, actual)