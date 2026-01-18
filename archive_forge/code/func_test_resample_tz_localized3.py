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
def test_resample_tz_localized3(self):
    rng = date_range('1/1/2011', periods=20000, freq='h')
    rng = rng.tz_localize('EST')
    ts = DataFrame(index=rng)
    ts['first'] = np.random.default_rng(2).standard_normal(len(rng))
    ts['second'] = np.cumsum(np.random.default_rng(2).standard_normal(len(rng)))
    expected = DataFrame({'first': ts.resample('YE').sum()['first'], 'second': ts.resample('YE').mean()['second']}, columns=['first', 'second'])
    result = ts.resample('YE').agg({'first': 'sum', 'second': 'mean'}).reindex(columns=['first', 'second'])
    tm.assert_frame_equal(result, expected)