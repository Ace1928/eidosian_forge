from __future__ import absolute_import, print_function, division
from datetime import datetime
import pytest
from petl.errors import FieldSelectionError
from petl.test.helpers import ieq
from petl.transform.reshape import melt, recast, transpose, pivot, flatten, \
from petl.transform.regex import split, capture
def test_recast4():
    table = (('id', 'variable', 'value'), (1, 'gender', 'F'), (2, 'age', 17), (1, 'age', 12), (3, 'gender', 'M'))
    result = recast(table, key='id')
    expect = (('id', 'age', 'gender'), (1, 12, 'F'), (2, 17, None), (3, None, 'M'))
    ieq(expect, result)