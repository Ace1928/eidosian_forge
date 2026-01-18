import os
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.api.types import CategoricalDtype
from pandas.tseries.offsets import Day
@pytest.mark.parametrize('ser', [Series(DatetimeIndex(['20180101', NaT, '20180103'])), Series(TimedeltaIndex(['0 days', NaT, '2 days']))], ids=lambda x: str(x.dtype))
def test_qcut_nat(ser, unit):
    ser = ser.dt.as_unit(unit)
    td = Timedelta(1, unit=unit).as_unit(unit)
    left = Series([ser[0] - td, np.nan, ser[2] - Day()], dtype=ser.dtype)
    right = Series([ser[2] - Day(), np.nan, ser[2]], dtype=ser.dtype)
    intervals = IntervalIndex.from_arrays(left, right)
    expected = Series(Categorical(intervals, ordered=True))
    result = qcut(ser, 2)
    tm.assert_series_equal(result, expected)