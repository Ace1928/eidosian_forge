from __future__ import absolute_import, print_function, division
import logging
import pytest
import petl as etl
from petl.transform.validation import validate
from petl.test.helpers import ieq
from petl.errors import FieldSelectionError
def test_row_length():
    table = (('foo', 'bar', 'baz'), (1, '2000-01-01', 'Y'), ('x', '2010-10-10'), (2, '2000/01/01', 'Y', True))
    expect = (('name', 'row', 'field', 'value', 'error'), ('__len__', 2, None, 2, 'AssertionError'), ('__len__', 3, None, 4, 'AssertionError'))
    actual = validate(table)
    debug(actual)
    ieq(expect, actual)
    ieq(expect, actual)