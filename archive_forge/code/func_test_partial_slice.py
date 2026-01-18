from datetime import datetime
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_partial_slice(self):
    rng = date_range(freq='D', start=datetime(2005, 1, 1), periods=500)
    s = Series(np.arange(len(rng)), index=rng)
    result = s['2005-05':'2006-02']
    expected = s['20050501':'20060228']
    tm.assert_series_equal(result, expected)
    result = s['2005-05':]
    expected = s['20050501':]
    tm.assert_series_equal(result, expected)
    result = s[:'2006-02']
    expected = s[:'20060228']
    tm.assert_series_equal(result, expected)
    result = s['2005-1-1']
    assert result == s.iloc[0]
    with pytest.raises(KeyError, match="^'2004-12-31'$"):
        s['2004-12-31']