from datetime import datetime
import re
import numpy as np
import pytest
from pandas._libs import lib
from pandas.errors import UnsupportedFunctionCall
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.indexes.datetimes import date_range
def test_resample_agg_readonly():
    index = date_range('2020-01-01', '2020-01-02', freq='1h')
    arr = np.zeros_like(index)
    arr.setflags(write=False)
    ser = Series(arr, index=index)
    rs = ser.resample('1D')
    expected = Series([pd.Timestamp(0), pd.Timestamp(0)], index=index[::24])
    result = rs.agg('last')
    tm.assert_series_equal(result, expected)
    result = rs.agg('first')
    tm.assert_series_equal(result, expected)
    result = rs.agg('max')
    tm.assert_series_equal(result, expected)
    result = rs.agg('min')
    tm.assert_series_equal(result, expected)