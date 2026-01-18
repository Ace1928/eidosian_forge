import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.api.types import (
from pandas.core.arrays import BooleanArray
from pandas.core.indexes.datetimelike import DatetimeIndexOpsMixin
@pytest.mark.parametrize('func', [np.bitwise_and, np.bitwise_or, np.bitwise_xor])
def test_numpy_ufuncs_bitwise(func):
    idx1 = Index([1, 2, 3, 4], dtype='int64')
    idx2 = Index([3, 4, 5, 6], dtype='int64')
    with tm.assert_produces_warning(None):
        result = func(idx1, idx2)
    expected = Index(func(idx1.values, idx2.values))
    tm.assert_index_equal(result, expected)