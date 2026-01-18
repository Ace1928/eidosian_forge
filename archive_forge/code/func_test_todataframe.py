from __future__ import division, print_function, absolute_import
import pytest
import petl as etl
from petl.test.helpers import ieq
from petl.io.pandas import todataframe, fromdataframe
def test_todataframe():
    tbl = [('foo', 'bar', 'baz'), ('apples', 1, 2.5), ('oranges', 3, 4.4), ('pears', 7, 0.1)]
    expect = pd.DataFrame.from_records(tbl[1:], columns=tbl[0])
    actual = todataframe(tbl)
    assert expect.equals(actual)