import numpy as np
import pytest
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_shift_fill_value(self, frame_or_series):
    dti = date_range('1/1/2000', periods=5, freq='h')
    ts = frame_or_series([1.0, 2.0, 3.0, 4.0, 5.0], index=dti)
    exp = frame_or_series([0.0, 1.0, 2.0, 3.0, 4.0], index=dti)
    result = ts.shift(1, fill_value=0.0)
    tm.assert_equal(result, exp)
    exp = frame_or_series([0.0, 0.0, 1.0, 2.0, 3.0], index=dti)
    result = ts.shift(2, fill_value=0.0)
    tm.assert_equal(result, exp)
    ts = frame_or_series([1, 2, 3])
    res = ts.shift(2, fill_value=0)
    assert tm.get_dtype(res) == tm.get_dtype(ts)
    obj = frame_or_series([1, 2, 3, 4, 5], index=dti)
    exp = frame_or_series([0, 1, 2, 3, 4], index=dti)
    result = obj.shift(1, fill_value=0)
    tm.assert_equal(result, exp)
    exp = frame_or_series([0, 0, 1, 2, 3], index=dti)
    result = obj.shift(2, fill_value=0)
    tm.assert_equal(result, exp)