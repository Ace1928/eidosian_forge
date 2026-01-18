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
@pytest.mark.parametrize('other', [1, np.int64(1), 1.0, np.timedelta64('NaT'), pd.Timedelta(days=2), 'invalid', np.arange(10, dtype='i8') * 24 * 3600 * 10 ** 9, np.arange(10).view('timedelta64[ns]') * 24 * 3600 * 10 ** 9, pd.Timestamp('2021-01-01').to_period('D')])
@pytest.mark.parametrize('index', [True, False])
def test_searchsorted_invalid_types(self, other, index):
    data = np.arange(10, dtype='i8') * 24 * 3600 * 10 ** 9
    arr = pd.DatetimeIndex(data, freq='D')._data
    if index:
        arr = pd.Index(arr)
    msg = '|'.join(['searchsorted requires compatible dtype or scalar', "value should be a 'Timestamp', 'NaT', or array of those. Got"])
    with pytest.raises(TypeError, match=msg):
        arr.searchsorted(other)