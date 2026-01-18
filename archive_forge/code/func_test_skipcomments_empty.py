from __future__ import absolute_import, print_function, division
import pytest
from petl.errors import FieldSelectionError
from petl.test.helpers import ieq
from petl.util import expr, empty, coalesce
from petl.transform.basics import cut, cat, addfield, rowslice, head, tail, \
def test_skipcomments_empty():
    table1 = (('##aaa', 'bbb', 'ccc'), ('##mmm',), ('#foo', 'bar'), ('##nnn', 1))
    table2 = skipcomments(table1, '##')
    expect2 = (('#foo', 'bar'),)
    ieq(expect2, table2)