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
def test_sub_datetime64_reso_mismatch(self):
    dti = pd.date_range('2016-01-01', periods=3)
    left = dti._data.as_unit('s')
    right = left.as_unit('ms')
    result = left - right
    exp_values = np.array([0, 0, 0], dtype='m8[ms]')
    expected = TimedeltaArray._simple_new(exp_values, dtype=exp_values.dtype)
    tm.assert_extension_array_equal(result, expected)
    result2 = right - left
    tm.assert_extension_array_equal(result2, expected)