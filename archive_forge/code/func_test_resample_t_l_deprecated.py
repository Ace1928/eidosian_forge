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
def test_resample_t_l_deprecated(self):
    msg_t = "'T' is deprecated and will be removed in a future version."
    msg_l = "'L' is deprecated and will be removed in a future version."
    with tm.assert_produces_warning(FutureWarning, match=msg_l):
        rng_l = period_range('2020-01-01 00:00:00 00:00', '2020-01-01 00:00:00 00:01', freq='L')
    ser = Series(np.arange(len(rng_l)), index=rng_l)
    rng = period_range('2020-01-01 00:00:00 00:00', '2020-01-01 00:00:00 00:01', freq='min')
    expected = Series([29999.5, 60000.0], index=rng)
    with tm.assert_produces_warning(FutureWarning, match=msg_t):
        result = ser.resample('T').mean()
    tm.assert_series_equal(result, expected)