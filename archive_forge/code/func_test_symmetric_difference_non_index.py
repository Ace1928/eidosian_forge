from datetime import datetime
import operator
import numpy as np
import pytest
from pandas._libs import lib
from pandas.core.dtypes.cast import find_common_type
from pandas import (
import pandas._testing as tm
from pandas.api.types import (
def test_symmetric_difference_non_index(self, sort):
    index1 = Index([1, 2, 3, 4], name='index1')
    index2 = np.array([2, 3, 4, 5])
    expected = Index([1, 5], name='index1')
    result = index1.symmetric_difference(index2, sort=sort)
    if sort in (None, True):
        tm.assert_index_equal(result, expected)
    else:
        tm.assert_index_equal(result.sort_values(), expected)
    assert result.name == 'index1'
    result = index1.symmetric_difference(index2, result_name='new_name', sort=sort)
    expected.name = 'new_name'
    if sort in (None, True):
        tm.assert_index_equal(result, expected)
    else:
        tm.assert_index_equal(result.sort_values(), expected)
    assert result.name == 'new_name'