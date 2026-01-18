from datetime import datetime
import operator
import numpy as np
import pytest
from pandas._libs import lib
from pandas.core.dtypes.cast import find_common_type
from pandas import (
import pandas._testing as tm
from pandas.api.types import (
@pytest.mark.parametrize('fname, sname, expected_name', [('A', 'A', 'A'), ('A', 'B', None), ('A', None, None), (None, 'B', None), (None, None, None)])
def test_union_unequal(self, index_flat, fname, sname, expected_name):
    if not index_flat.is_unique:
        index = index_flat.unique()
    else:
        index = index_flat
    first = index.copy().set_names(fname)
    second = index[1:].set_names(sname)
    union = first.union(second).sort_values()
    expected = index.set_names(expected_name).sort_values()
    tm.assert_index_equal(union, expected)