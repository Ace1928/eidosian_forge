import numpy as np
import pytest
from pandas._libs import index as libindex
from pandas.compat import IS64
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_setitem_interval_with_slice(self):
    ii = IntervalIndex.from_breaks(range(4, 15))
    ser = Series(range(10), index=ii)
    orig = ser.copy()
    ser.loc[1:3] = 20
    tm.assert_series_equal(ser, orig)
    ser.loc[6:8] = 19
    orig.iloc[1:4] = 19
    tm.assert_series_equal(ser, orig)
    ser2 = Series(range(5), index=ii[::2])
    orig2 = ser2.copy()
    ser2.loc[6:8] = 22
    orig2.iloc[1] = 22
    tm.assert_series_equal(ser2, orig2)
    ser2.loc[5:7] = 21
    orig2.iloc[:2] = 21
    tm.assert_series_equal(ser2, orig2)