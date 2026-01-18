from datetime import (
from decimal import Decimal
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core import nanops
from pandas.core.arrays.string_arrow import ArrowStringArrayNumpySemantics
def test_timedelta64_analytics(self):
    dti = date_range('2012-1-1', periods=3, freq='D')
    td = Series(dti) - Timestamp('20120101')
    result = td.idxmin()
    assert result == 0
    result = td.idxmax()
    assert result == 2
    td[0] = np.nan
    result = td.idxmin()
    assert result == 1
    result = td.idxmax()
    assert result == 2
    s1 = Series(date_range('20120101', periods=3))
    s2 = Series(date_range('20120102', periods=3))
    expected = Series(s2 - s1)
    result = np.abs(s1 - s2)
    tm.assert_series_equal(result, expected)
    result = (s1 - s2).abs()
    tm.assert_series_equal(result, expected)
    result = td.max()
    expected = Timedelta('2 days')
    assert result == expected
    result = td.min()
    expected = Timedelta('1 days')
    assert result == expected