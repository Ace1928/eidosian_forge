from __future__ import absolute_import, print_function, division
import pytest
from petl.errors import FieldSelectionError
from petl.test.failonerror import assert_failonerror
from petl.test.helpers import ieq
from petl.transform.conversions import convert, convertall, convertnumbers, \
from functools import partial
def test_replace_unhashable():
    table1 = (('foo', 'bar'), ('a', ['b']), ('c', None))
    expect = (('foo', 'bar'), ('a', ['b']), ('c', []))
    actual = replace(table1, 'bar', None, [])
    ieq(expect, actual)