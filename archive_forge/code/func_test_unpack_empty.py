from __future__ import absolute_import, print_function, division
import pytest
from petl.errors import ArgumentError
from petl.test.helpers import ieq
from petl.transform.unpacks import unpack, unpackdict
def test_unpack_empty():
    table1 = (('foo', 'bar'),)
    table2 = unpack(table1, 'bar', ['baz', 'quux'])
    expect2 = (('foo', 'baz', 'quux'),)
    ieq(expect2, table2)