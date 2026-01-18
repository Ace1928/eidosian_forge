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
@pytest.mark.parametrize('dates', [dates1, dates2, dates3])
def test_resample_timegrouper2(dates, unit):
    dates = DatetimeIndex(dates).as_unit(unit)
    df = DataFrame({'A': dates, 'B': np.arange(len(dates)), 'C': np.arange(len(dates))})
    result = df.set_index('A').resample('ME').count()
    exp_idx = DatetimeIndex(['2014-07-31', '2014-08-31', '2014-09-30', '2014-10-31', '2014-11-30'], freq='ME', name='A').as_unit(unit)
    expected = DataFrame({'B': [1, 0, 2, 2, 1], 'C': [1, 0, 2, 2, 1]}, index=exp_idx, columns=['B', 'C'])
    if df['A'].isna().any():
        expected.index = expected.index._with_freq(None)
    tm.assert_frame_equal(result, expected)
    result = df.groupby(Grouper(freq='ME', key='A')).count()
    tm.assert_frame_equal(result, expected)