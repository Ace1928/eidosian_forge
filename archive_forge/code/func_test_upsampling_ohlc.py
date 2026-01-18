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
@pytest.mark.parametrize('freq, period_mult', [('h', 24), ('12h', 2)])
@pytest.mark.parametrize('kind', [None, 'period'])
def test_upsampling_ohlc(self, freq, period_mult, kind):
    pi = period_range(start='2000', freq='D', periods=10)
    s = Series(range(len(pi)), index=pi)
    expected = s.to_timestamp().resample(freq).ohlc().to_period(freq)
    new_index = period_range(start='2000', freq=freq, periods=period_mult * len(pi))
    expected = expected.reindex(new_index)
    msg = "The 'kind' keyword in Series.resample is deprecated"
    with tm.assert_produces_warning(FutureWarning, match=msg):
        result = s.resample(freq, kind=kind).ohlc()
    tm.assert_frame_equal(result, expected)