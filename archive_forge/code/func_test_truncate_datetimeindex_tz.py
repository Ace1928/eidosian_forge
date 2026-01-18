from datetime import datetime
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_truncate_datetimeindex_tz(self):
    idx = date_range('4/1/2005', '4/30/2005', freq='D', tz='US/Pacific')
    s = Series(range(len(idx)), index=idx)
    with pytest.raises(TypeError, match='Cannot compare tz-naive'):
        s.truncate(datetime(2005, 4, 2), datetime(2005, 4, 4))
    lb = idx[1]
    ub = idx[3]
    result = s.truncate(lb.to_pydatetime(), ub.to_pydatetime())
    expected = Series([1, 2, 3], index=idx[1:4])
    tm.assert_series_equal(result, expected)