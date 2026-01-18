from datetime import datetime
from hypothesis import given
import numpy as np
import pytest
from pandas.core.dtypes.common import is_scalar
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas._testing._hypothesis import OPTIONAL_ONE_OF_ALL
def test_where_interval_noop(self):
    df = DataFrame([pd.Interval(0, 0)])
    res = df.where(df.notna())
    tm.assert_frame_equal(res, df)
    ser = df[0]
    res = ser.where(ser.notna())
    tm.assert_series_equal(res, ser)