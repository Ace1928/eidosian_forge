from __future__ import absolute_import, print_function, division
from datetime import datetime
import pytest
from petl.errors import FieldSelectionError
from petl.test.helpers import ieq
from petl.transform.reshape import melt, recast, transpose, pivot, flatten, \
from petl.transform.regex import split, capture
def test_pivot_headerless():
    table1 = []
    with pytest.raises(FieldSelectionError):
        for i in pivot(table1, 'region', 'gender', 'units', sum):
            pass