from __future__ import absolute_import, print_function, division
import pytest
from petl.errors import FieldSelectionError
from petl.test.helpers import ieq
from petl.util import expr, empty, coalesce
from petl.transform.basics import cut, cat, addfield, rowslice, head, tail, \
def test_stack_dupfields():
    table1 = (('foo', 'foo'), (1, 'A'), (2,), (3, 'B', True))
    actual = stack(table1)
    expect = (('foo', 'foo'), (1, 'A'), (2, None), (3, 'B'))
    ieq(expect, actual)
    table2 = (('foo', 'foo', 'bar'), (4, 'C', True), (5, 'D', False))
    actual = stack(table1, table2)
    expect = (('foo', 'foo'), (1, 'A'), (2, None), (3, 'B'), (4, 'C'), (5, 'D'))
    ieq(expect, actual)