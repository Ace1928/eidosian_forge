from datetime import datetime
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_truncate_periodindex(self):
    idx1 = pd.PeriodIndex([pd.Period('2017-09-02'), pd.Period('2017-09-02'), pd.Period('2017-09-03')])
    series1 = Series([1, 2, 3], index=idx1)
    result1 = series1.truncate(after='2017-09-02')
    expected_idx1 = pd.PeriodIndex([pd.Period('2017-09-02'), pd.Period('2017-09-02')])
    tm.assert_series_equal(result1, Series([1, 2], index=expected_idx1))
    idx2 = pd.PeriodIndex([pd.Period('2017-09-03'), pd.Period('2017-09-02'), pd.Period('2017-09-03')])
    series2 = Series([1, 2, 3], index=idx2)
    result2 = series2.sort_index().truncate(after='2017-09-02')
    expected_idx2 = pd.PeriodIndex([pd.Period('2017-09-02')])
    tm.assert_series_equal(result2, Series([2], index=expected_idx2))