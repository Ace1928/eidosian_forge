from __future__ import absolute_import, print_function, division
import pytest
from petl.test.helpers import ieq
from petl.errors import FieldSelectionError
from petl.util import fieldnames
from petl.transform.headers import setheader, extendheader, pushheader, skip, \
def test_prefixheader_headerless():
    table = []
    actual = prefixheader(table, 'pre_')
    expect = []
    ieq(expect, actual)