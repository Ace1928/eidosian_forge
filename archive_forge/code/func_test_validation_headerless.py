from __future__ import absolute_import, print_function, division
import logging
import pytest
import petl as etl
from petl.transform.validation import validate
from petl.test.helpers import ieq
from petl.errors import FieldSelectionError
def test_validation_headerless():
    header = ('foo', 'bar', 'baz')
    table = []
    expect = (('name', 'row', 'field', 'value', 'error'), ('__header__', 0, None, None, 'AssertionError'))
    actual = validate(table, header=header)
    ieq(expect, actual)
    ieq(expect, actual)