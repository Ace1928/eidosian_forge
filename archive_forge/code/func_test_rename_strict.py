from __future__ import absolute_import, print_function, division
import pytest
from petl.test.helpers import ieq
from petl.errors import FieldSelectionError
from petl.util import fieldnames
from petl.transform.headers import setheader, extendheader, pushheader, skip, \
def test_rename_strict():
    table = (('foo', 'bar'), ('M', 12), ('F', 34), ('-', 56))
    result = rename(table, 'baz', 'quux')
    try:
        fieldnames(result)
    except FieldSelectionError:
        pass
    else:
        assert False, 'exception expected'
    result = rename(table, 2, 'quux')
    try:
        fieldnames(result)
    except FieldSelectionError:
        pass
    else:
        assert False, 'exception expected'
    result = rename(table, 'baz', 'quux', strict=False)
    assert fieldnames(result) == ('foo', 'bar')
    result = rename(table, 2, 'quux', strict=False)
    assert fieldnames(result) == ('foo', 'bar')