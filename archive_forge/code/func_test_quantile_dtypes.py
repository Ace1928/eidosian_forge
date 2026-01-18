import numpy as np
import pytest
from pandas.core.dtypes.common import is_integer
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.indexes.datetimes import Timestamp
@pytest.mark.parametrize('dtype', [int, float, 'Int64'])
def test_quantile_dtypes(self, dtype):
    result = Series([1, 2, 3], dtype=dtype).quantile(np.arange(0, 1, 0.25))
    expected = Series(np.arange(1, 3, 0.5), index=np.arange(0, 1, 0.25))
    if dtype == 'Int64':
        expected = expected.astype('Float64')
    tm.assert_series_equal(result, expected)