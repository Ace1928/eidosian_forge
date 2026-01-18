import numpy as np
from pandas import (
import pandas._testing as tm
def test_tdi_division(self, index_or_series):
    scalar = Timedelta(days=31)
    td = index_or_series([scalar, scalar, scalar + Timedelta(minutes=5, seconds=3), NaT], dtype='m8[ns]')
    result = td / np.timedelta64(1, 'D')
    expected = index_or_series([31, 31, (31 * 86400 + 5 * 60 + 3) / 86400.0, np.nan])
    tm.assert_equal(result, expected)
    result = td / np.timedelta64(1, 's')
    expected = index_or_series([31 * 86400, 31 * 86400, 31 * 86400 + 5 * 60 + 3, np.nan])
    tm.assert_equal(result, expected)