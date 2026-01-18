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
def test_quarterly_resampling(self):
    rng = period_range('2000Q1', periods=10, freq='Q-DEC')
    ts = Series(np.arange(10), index=rng)
    result = ts.resample('Y').mean()
    exp = ts.to_timestamp().resample('YE').mean().to_period()
    tm.assert_series_equal(result, exp)