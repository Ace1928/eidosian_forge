import numpy as np
import pytest
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_shift_dst(self, frame_or_series):
    dates = date_range('2016-11-06', freq='h', periods=10, tz='US/Eastern')
    obj = frame_or_series(dates)
    res = obj.shift(0)
    tm.assert_equal(res, obj)
    assert tm.get_dtype(res) == 'datetime64[ns, US/Eastern]'
    res = obj.shift(1)
    exp_vals = [NaT] + dates.astype(object).values.tolist()[:9]
    exp = frame_or_series(exp_vals)
    tm.assert_equal(res, exp)
    assert tm.get_dtype(res) == 'datetime64[ns, US/Eastern]'
    res = obj.shift(-2)
    exp_vals = dates.astype(object).values.tolist()[2:] + [NaT, NaT]
    exp = frame_or_series(exp_vals)
    tm.assert_equal(res, exp)
    assert tm.get_dtype(res) == 'datetime64[ns, US/Eastern]'