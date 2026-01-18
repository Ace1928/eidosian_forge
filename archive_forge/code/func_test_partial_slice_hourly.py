from datetime import datetime
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_partial_slice_hourly(self):
    rng = date_range(freq='min', start=datetime(2005, 1, 1, 20, 0, 0), periods=500)
    s = Series(np.arange(len(rng)), index=rng)
    result = s['2005-1-1']
    tm.assert_series_equal(result, s.iloc[:60 * 4])
    result = s['2005-1-1 20']
    tm.assert_series_equal(result, s.iloc[:60])
    assert s['2005-1-1 20:00'] == s.iloc[0]
    with pytest.raises(KeyError, match="^'2004-12-31 00:15'$"):
        s['2004-12-31 00:15']