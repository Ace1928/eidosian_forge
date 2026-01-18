from datetime import datetime
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_truncate_one_element_series(self):
    series = Series([0.1], index=pd.DatetimeIndex(['2020-08-04']))
    before = pd.Timestamp('2020-08-02')
    after = pd.Timestamp('2020-08-04')
    result = series.truncate(before=before, after=after)
    tm.assert_series_equal(result, series)