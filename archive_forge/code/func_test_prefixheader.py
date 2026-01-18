from __future__ import absolute_import, print_function, division
import pytest
from petl.test.helpers import ieq
from petl.errors import FieldSelectionError
from petl.util import fieldnames
from petl.transform.headers import setheader, extendheader, pushheader, skip, \
def test_prefixheader():
    table1 = (('foo', 'bar'), (1, 'A'), (2, 'B'))
    expect = (('pre_foo', 'pre_bar'), (1, 'A'), (2, 'B'))
    actual = prefixheader(table1, 'pre_')
    ieq(expect, actual)
    ieq(expect, actual)