from __future__ import absolute_import, print_function, division
import pytest
from petl.errors import FieldSelectionError
from petl.test.failonerror import assert_failonerror
from petl.test.helpers import ieq
from petl.transform.conversions import convert, convertall, convertnumbers, \
from functools import partial
def test_replace_where():
    tbl1 = (('foo', 'bar'), ('a', 1), ('b', 2))
    expect = (('foo', 'bar'), ('a', 1), ('b', 4))
    actual = replace(tbl1, 'bar', 2, 4, where=lambda r: r.foo == 'b')
    ieq(expect, actual)
    ieq(expect, actual)
    actual = replace(tbl1, 'bar', 2, 4, where="{foo} == 'b'")
    ieq(expect, actual)
    ieq(expect, actual)