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
def test_resample_weekly_all_na(self):
    rng = date_range('1/1/2000', periods=10, freq='W-WED')
    ts = Series(np.random.default_rng(2).standard_normal(len(rng)), index=rng)
    result = ts.resample('W-THU').asfreq()
    assert result.isna().all()
    result = ts.resample('W-THU').asfreq().ffill()[:-1]
    expected = ts.asfreq('W-THU').ffill()
    tm.assert_series_equal(result, expected)