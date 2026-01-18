import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_pct_change_with_duplicate_axis(self):
    common_idx = date_range('2019-11-14', periods=5, freq='D')
    result = Series(range(5), common_idx).pct_change(freq='B')
    expected = Series([np.nan, np.inf, np.nan, np.nan, 3.0], common_idx)
    tm.assert_series_equal(result, expected)