from datetime import timedelta
import numpy as np
import pytest
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.indexes.timedeltas import timedelta_range
def test_resample_timedelta_values():
    times = timedelta_range('1 day', '6 day', freq='4D')
    df = DataFrame({'time': times}, index=times)
    times2 = timedelta_range('1 day', '6 day', freq='2D')
    exp = Series(times2, index=times2, name='time')
    exp.iloc[1] = pd.NaT
    res = df.resample('2D').first()['time']
    tm.assert_series_equal(res, exp)
    res = df['time'].resample('2D').first()
    tm.assert_series_equal(res, exp)