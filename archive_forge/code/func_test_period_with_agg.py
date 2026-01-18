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
def test_period_with_agg():
    s2 = Series(np.random.default_rng(2).integers(0, 5, 50), index=period_range('2012-01-01', freq='h', periods=50), dtype='float64')
    expected = s2.to_timestamp().resample('D').mean().to_period()
    msg = 'Resampling with a PeriodIndex is deprecated'
    with tm.assert_produces_warning(FutureWarning, match=msg):
        rs = s2.resample('D')
    result = rs.agg(lambda x: x.mean())
    tm.assert_series_equal(result, expected)