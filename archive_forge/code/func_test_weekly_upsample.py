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
@pytest.mark.parametrize('day', DAYS)
@pytest.mark.parametrize('target', ['D', 'B'])
@pytest.mark.parametrize('convention', ['start', 'end'])
def test_weekly_upsample(self, day, target, convention, simple_period_range_series):
    freq = f'W-{day}'
    ts = simple_period_range_series('1/1/1990', '12/31/1995', freq=freq)
    warn = None if target == 'D' else FutureWarning
    msg = 'PeriodDtype\\[B\\] is deprecated'
    if warn is None:
        msg = 'Resampling with a PeriodIndex is deprecated'
        warn = FutureWarning
    with tm.assert_produces_warning(warn, match=msg):
        result = ts.resample(target, convention=convention).ffill()
        expected = result.to_timestamp(target, how=convention)
        expected = expected.asfreq(target, 'ffill').to_period()
    tm.assert_series_equal(result, expected)