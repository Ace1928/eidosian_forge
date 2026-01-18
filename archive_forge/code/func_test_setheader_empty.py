from __future__ import absolute_import, print_function, division
import pytest
from petl.test.helpers import ieq
from petl.errors import FieldSelectionError
from petl.util import fieldnames
from petl.transform.headers import setheader, extendheader, pushheader, skip, \
def test_setheader_empty():
    table1 = (('foo', 'bar'),)
    table2 = setheader(table1, ['foofoo', 'barbar'])
    expect2 = (('foofoo', 'barbar'),)
    ieq(expect2, table2)