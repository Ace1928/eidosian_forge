from datetime import datetime
import warnings
import dateutil
import numpy as np
import pytest
import pytz
from pandas._libs.tslibs.ccalendar import (
from pandas._libs.tslibs.period import IncompatibleFrequency
from pandas.errors import InvalidIndexError
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.indexes.datetimes import date_range
from pandas.core.indexes.period import (
from pandas.core.resample import _get_period_range_edges
from pandas.tseries import offsets
def test_resample_with_offset_month(self):
    pi = period_range('19910905 12:00', '19910909 1:00', freq='h')
    ser = Series(np.arange(len(pi)), index=pi)
    msg = 'Resampling with a PeriodIndex is deprecated'
    with tm.assert_produces_warning(FutureWarning, match=msg):
        rs = ser.resample('M', offset='3h')
    result = rs.mean()
    result = result.to_timestamp('M')
    expected = ser.to_timestamp().resample('ME', offset='3h').mean()
    expected.index = expected.index._with_freq(None)
    tm.assert_series_equal(result, expected)