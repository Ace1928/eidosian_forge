from __future__ import absolute_import, print_function, division
import pytest
from petl.compat import next
from petl.errors import ArgumentError
from petl.test.helpers import ieq, eq_
from petl.transform.regex import capture, split, search, searchcomplement, splitdown
from petl.transform.basics import TransformError
def test_search_2():
    table = (('foo', 'bar', 'baz'), ('aa', 4, 9.3), ('aaa', 2, 88.2), ('b', 1, 23.3), ('ccc', 8, 42.0), ('bb', 7, 100.9), ('c', 2))
    actual = search(table, 'foo', '[ab]{2}')
    expect = (('foo', 'bar', 'baz'), ('aa', 4, 9.3), ('aaa', 2, 88.2), ('bb', 7, 100.9))
    ieq(expect, actual)
    ieq(expect, actual)