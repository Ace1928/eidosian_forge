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
def test_resample_apply_with_additional_args2():

    def f(data, add_arg):
        return np.mean(data) * add_arg
    multiplier = 10
    df = DataFrame({'A': 1, 'B': 2}, index=date_range('2017', periods=10))
    msg = 'DataFrameGroupBy.resample operated on the grouping columns'
    with tm.assert_produces_warning(DeprecationWarning, match=msg):
        result = df.groupby('A').resample('D').agg(f, multiplier).astype(float)
    msg = 'DataFrameGroupBy.resample operated on the grouping columns'
    with tm.assert_produces_warning(DeprecationWarning, match=msg):
        expected = df.groupby('A').resample('D').mean().multiply(multiplier)
    tm.assert_frame_equal(result, expected)