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
def test_repeat_preserves_tz(self):
    dti = pd.date_range('2000', periods=2, freq='D', tz='US/Central')
    arr = dti._data
    repeated = arr.repeat([1, 1])
    expected = DatetimeArray._from_sequence(arr.asi8, dtype=arr.dtype)
    tm.assert_equal(repeated, expected)