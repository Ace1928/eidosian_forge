from __future__ import absolute_import, print_function, division
from petl.compat import PY2
from petl.test.helpers import ieq, eq_
from petl.util.counting import valuecount, valuecounter, valuecounts, \
def test_valuecount():
    table = (('foo', 'bar'), ('a', 1), ('b', 2), ('b', 7))
    n, f = valuecount(table, 'foo', 'b')
    eq_(2, n)
    eq_(2.0 / 3, f)