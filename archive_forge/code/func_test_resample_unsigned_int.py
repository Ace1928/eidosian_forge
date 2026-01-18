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
def test_resample_unsigned_int(any_unsigned_int_numpy_dtype, unit):
    df = DataFrame(index=date_range(start='2000-01-01', end='2000-01-03 23', freq='12h').as_unit(unit), columns=['x'], data=[0, 1, 0] * 2, dtype=any_unsigned_int_numpy_dtype)
    df = df.loc[(df.index < '2000-01-02') | (df.index > '2000-01-03'), :]
    result = df.resample('D').max()
    expected = DataFrame([1, np.nan, 0], columns=['x'], index=date_range(start='2000-01-01', end='2000-01-03 23', freq='D').as_unit(unit))
    tm.assert_frame_equal(result, expected)