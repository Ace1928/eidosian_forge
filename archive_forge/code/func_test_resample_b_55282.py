from datetime import datetime
from functools import partial
import numpy as np
import pytest
import pytz
from pandas._libs import lib
from pandas._typing import DatetimeNaTType
from pandas.compat import is_platform_windows
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.groupby.grouper import Grouper
from pandas.core.indexes.datetimes import date_range
from pandas.core.indexes.period import (
from pandas.core.resample import (
from pandas.tseries import offsets
from pandas.tseries.offsets import Minute
def test_resample_b_55282(unit):
    dti = date_range('2023-09-26', periods=6, freq='12h', unit=unit)
    ser = Series([1, 2, 3, 4, 5, 6], index=dti)
    result = ser.resample('B', closed='right', label='right').mean()
    exp_dti = DatetimeIndex([datetime(2023, 9, 26), datetime(2023, 9, 27), datetime(2023, 9, 28), datetime(2023, 9, 29)], freq='B').as_unit(unit)
    expected = Series([1.0, 2.5, 4.5, 6.0], index=exp_dti)
    tm.assert_series_equal(result, expected)