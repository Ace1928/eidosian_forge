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
def test_resample_dup_index():
    df = DataFrame(np.random.default_rng(2).standard_normal((4, 12)), index=[2000, 2000, 2000, 2000], columns=[Period(year=2000, month=i + 1, freq='M') for i in range(12)])
    df.iloc[3, :] = np.nan
    warning_msg = 'DataFrame.resample with axis=1 is deprecated.'
    with tm.assert_produces_warning(FutureWarning, match=warning_msg):
        result = df.resample('QE', axis=1).mean()
    msg = 'DataFrame.groupby with axis=1 is deprecated'
    with tm.assert_produces_warning(FutureWarning, match=msg):
        expected = df.groupby(lambda x: int((x.month - 1) / 3), axis=1).mean()
    expected.columns = [Period(year=2000, quarter=i + 1, freq='Q') for i in range(4)]
    tm.assert_frame_equal(result, expected)