from __future__ import absolute_import, print_function, division
import pytest
from petl.test.helpers import ieq
from petl.errors import FieldSelectionError
from petl.util import fieldnames
from petl.transform.headers import setheader, extendheader, pushheader, skip, \
def test_pushheader():
    table1 = (('a', 1), ('b', 2))
    table2 = pushheader(table1, ['foo', 'bar'])
    expect2 = (('foo', 'bar'), ('a', 1), ('b', 2))
    ieq(expect2, table2)
    ieq(expect2, table2)