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
@pytest.mark.parametrize('index', [True, False])
def test_searchsorted_tzawareness_compat(self, index):
    data = np.arange(10, dtype='i8') * 24 * 3600 * 10 ** 9
    arr = pd.DatetimeIndex(data, freq='D')._data
    if index:
        arr = pd.Index(arr)
    mismatch = arr.tz_localize('Asia/Tokyo')
    msg = 'Cannot compare tz-naive and tz-aware datetime-like objects'
    with pytest.raises(TypeError, match=msg):
        arr.searchsorted(mismatch[0])
    with pytest.raises(TypeError, match=msg):
        arr.searchsorted(mismatch)
    with pytest.raises(TypeError, match=msg):
        mismatch.searchsorted(arr[0])
    with pytest.raises(TypeError, match=msg):
        mismatch.searchsorted(arr)