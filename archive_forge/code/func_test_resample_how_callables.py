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
def test_resample_how_callables(unit):
    data = np.arange(5, dtype=np.int64)
    ind = date_range(start='2014-01-01', periods=len(data), freq='d').as_unit(unit)
    df = DataFrame({'A': data, 'B': data}, index=ind)

    def fn(x, a=1):
        return str(type(x))

    class FnClass:

        def __call__(self, x):
            return str(type(x))
    df_standard = df.resample('ME').apply(fn)
    df_lambda = df.resample('ME').apply(lambda x: str(type(x)))
    df_partial = df.resample('ME').apply(partial(fn))
    df_partial2 = df.resample('ME').apply(partial(fn, a=2))
    df_class = df.resample('ME').apply(FnClass())
    tm.assert_frame_equal(df_standard, df_lambda)
    tm.assert_frame_equal(df_standard, df_partial)
    tm.assert_frame_equal(df_standard, df_partial2)
    tm.assert_frame_equal(df_standard, df_class)