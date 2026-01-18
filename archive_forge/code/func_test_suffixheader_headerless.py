from __future__ import absolute_import, print_function, division
import pytest
from petl.test.helpers import ieq
from petl.errors import FieldSelectionError
from petl.util import fieldnames
from petl.transform.headers import setheader, extendheader, pushheader, skip, \
def test_suffixheader_headerless():
    table = []
    actual = suffixheader(table, '_suf')
    expect = []
    ieq(expect, actual)