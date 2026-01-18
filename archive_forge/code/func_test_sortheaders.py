from __future__ import absolute_import, print_function, division
import pytest
from petl.test.helpers import ieq
from petl.errors import FieldSelectionError
from petl.util import fieldnames
from petl.transform.headers import setheader, extendheader, pushheader, skip, \
def test_sortheaders():
    table1 = (('id', 'foo', 'bar', 'baz'), ('a', 1, 2, 3), ('b', 4, 5, 6))
    expect = (('bar', 'baz', 'foo', 'id'), (2, 3, 1, 'a'), (5, 6, 4, 'b'))
    actual = sortheader(table1)
    ieq(expect, actual)