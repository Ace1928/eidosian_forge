from __future__ import absolute_import, print_function, division
import pytest
from petl.compat import next
from petl.errors import ArgumentError
from petl.test.helpers import ieq, eq_
from petl.transform.regex import capture, split, search, searchcomplement, splitdown
from petl.transform.basics import TransformError
def test_splitdown():
    tbl = ((u'name', u'roles'), (u'Jane Doe', u'president,engineer,tailor,lawyer'), (u'John Doe', u'rocket scientist,optometrist,chef,knight,sailor'))
    actual = splitdown(tbl, 'roles', ',')
    expect = ((u'name', u'roles'), (u'Jane Doe', u'president'), (u'Jane Doe', u'engineer'), (u'Jane Doe', u'tailor'), (u'Jane Doe', u'lawyer'), (u'John Doe', u'rocket scientist'), (u'John Doe', u'optometrist'), (u'John Doe', u'chef'), (u'John Doe', u'knight'), (u'John Doe', u'sailor'))
    ieq(expect, actual)
    ieq(expect, actual)
    ieq(expect, actual)
    ieq(expect, actual)
    ieq(expect, actual)
    ieq(expect, actual)
    ieq(expect, actual)
    ieq(expect, actual)
    ieq(expect, actual)
    ieq(expect, actual)