from __future__ import absolute_import, print_function, division
import pytest
from petl.compat import next
from petl.errors import ArgumentError
from petl.test.helpers import ieq, eq_
from petl.transform.regex import capture, split, search, searchcomplement, splitdown
from petl.transform.basics import TransformError
def test_search_headerless():
    table = []
    actual = search(table, 'foo', '[ab]{2}')
    expect = []
    ieq(expect, actual)
    ieq(expect, actual)