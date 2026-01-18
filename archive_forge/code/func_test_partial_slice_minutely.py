from datetime import datetime
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_partial_slice_minutely(self):
    rng = date_range(freq='s', start=datetime(2005, 1, 1, 23, 59, 0), periods=500)
    s = Series(np.arange(len(rng)), index=rng)
    result = s['2005-1-1 23:59']
    tm.assert_series_equal(result, s.iloc[:60])
    result = s['2005-1-1']
    tm.assert_series_equal(result, s.iloc[:60])
    assert s[Timestamp('2005-1-1 23:59:00')] == s.iloc[0]
    with pytest.raises(KeyError, match="^'2004-12-31 00:00:00'$"):
        s['2004-12-31 00:00:00']