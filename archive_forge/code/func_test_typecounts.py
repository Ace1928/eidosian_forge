from __future__ import absolute_import, print_function, division
from petl.compat import PY2
from petl.test.helpers import ieq, eq_
from petl.util.counting import valuecount, valuecounter, valuecounts, \
def test_typecounts():
    table = (('foo', 'bar', 'baz'), (b'A', 1, 2.0), (b'B', u'2', 3.4), (u'B', u'3', 7.8, True), (b'D', u'xyz', 9.0), (b'E', 42))
    actual = typecounts(table, 'foo')
    if PY2:
        expect = (('type', 'count', 'frequency'), ('str', 4, 4.0 / 5), ('unicode', 1, 1.0 / 5))
    else:
        expect = (('type', 'count', 'frequency'), ('bytes', 4, 4.0 / 5), ('str', 1, 1.0 / 5))
    ieq(expect, actual)
    actual = typecounts(table, 'bar')
    if PY2:
        expect = (('type', 'count', 'frequency'), ('unicode', 3, 3.0 / 5), ('int', 2, 2.0 / 5))
    else:
        expect = (('type', 'count', 'frequency'), ('str', 3, 3.0 / 5), ('int', 2, 2.0 / 5))
    ieq(expect, actual)
    actual = typecounts(table, 'baz')
    expect = (('type', 'count', 'frequency'), ('float', 4, 4.0 / 5), ('NoneType', 1, 1.0 / 5))
    ieq(expect, actual)