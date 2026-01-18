from __future__ import absolute_import, print_function, division
from petl.compat import PY2
from petl.test.helpers import ieq, eq_
from petl.util.counting import valuecount, valuecounter, valuecounts, \
def test_valuecounter():
    table = (('foo', 'bar'), ('a', 1), ('b', 2), ('b', 7))
    actual = valuecounter(table, 'foo')
    expect = {'b': 2, 'a': 1}
    eq_(expect, actual)