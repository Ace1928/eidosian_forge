from __future__ import absolute_import, print_function, division
import pytest
from petl.test.helpers import ieq
from petl.errors import FieldSelectionError
from petl.util import fieldnames
from petl.transform.headers import setheader, extendheader, pushheader, skip, \
def test_skip_empty():
    table1 = (('#aaa', 'bbb', 'ccc'), ('#mmm',), ('foo', 'bar'))
    table2 = skip(table1, 2)
    expect2 = (('foo', 'bar'),)
    ieq(expect2, table2)