from __future__ import annotations
from datetime import timedelta
import operator
import numpy as np
import pytest
from pandas._libs.tslibs import tz_compare
from pandas.core.dtypes.dtypes import DatetimeTZDtype
import pandas as pd
import pandas._testing as tm
from pandas.core.arrays import (
def test_value_counts_preserves_tz(self):
    dti = pd.date_range('2000', periods=2, freq='D', tz='US/Central')
    arr = dti._data.repeat([4, 3])
    result = arr.value_counts()
    assert result.index.equals(dti)
    arr[-2] = pd.NaT
    result = arr.value_counts(dropna=False)
    expected = pd.Series([4, 2, 1], index=[dti[0], dti[1], pd.NaT], name='count')
    tm.assert_series_equal(result, expected)