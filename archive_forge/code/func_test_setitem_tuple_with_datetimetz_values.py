from datetime import (
from decimal import Decimal
import numpy as np
import pytest
from pandas.compat.numpy import np_version_gte1p24
from pandas.errors import IndexingError
from pandas.core.dtypes.common import is_list_like
from pandas import (
import pandas._testing as tm
from pandas.tseries.offsets import BDay
def test_setitem_tuple_with_datetimetz_values(self):
    arr = date_range('2017', periods=4, tz='US/Eastern')
    index = [(0, 1), (0, 2), (0, 3), (0, 4)]
    result = Series(arr, index=index)
    expected = result.copy()
    result[0, 1] = np.nan
    expected.iloc[0] = np.nan
    tm.assert_series_equal(result, expected)