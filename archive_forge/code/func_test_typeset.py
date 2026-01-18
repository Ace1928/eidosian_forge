from __future__ import absolute_import, print_function, division
from petl.test.helpers import eq_
from petl.compat import PY2
from petl.util.misc import typeset, diffvalues, diffheaders
def test_typeset():
    table = (('foo', 'bar', 'baz'), (b'A', 1, u'2'), (b'B', '2', u'3.4'), (b'B', '3', u'7.8', True), (u'D', u'xyz', 9.0), (b'E', 42))
    actual = typeset(table, 'foo')
    if PY2:
        expect = {'str', 'unicode'}
    else:
        expect = {'bytes', 'str'}
    eq_(expect, actual)