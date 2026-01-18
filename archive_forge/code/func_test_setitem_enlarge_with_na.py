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
@pytest.mark.parametrize('na, target_na, dtype, target_dtype, indexer, warn', [(NA, NA, 'Int64', 'Int64', 1, None), (NA, NA, 'Int64', 'Int64', 2, None), (NA, np.nan, 'int64', 'float64', 1, None), (NA, np.nan, 'int64', 'float64', 2, None), (NaT, NaT, 'int64', 'object', 1, FutureWarning), (NaT, NaT, 'int64', 'object', 2, None), (np.nan, NA, 'Int64', 'Int64', 1, None), (np.nan, NA, 'Int64', 'Int64', 2, None), (np.nan, NA, 'Float64', 'Float64', 1, None), (np.nan, NA, 'Float64', 'Float64', 2, None), (np.nan, np.nan, 'int64', 'float64', 1, None), (np.nan, np.nan, 'int64', 'float64', 2, None)])
def test_setitem_enlarge_with_na(self, na, target_na, dtype, target_dtype, indexer, warn):
    ser = Series([1, 2], dtype=dtype)
    with tm.assert_produces_warning(warn, match='incompatible dtype'):
        ser[indexer] = na
    expected_values = [1, target_na] if indexer == 1 else [1, 2, target_na]
    expected = Series(expected_values, dtype=target_dtype)
    tm.assert_series_equal(ser, expected)