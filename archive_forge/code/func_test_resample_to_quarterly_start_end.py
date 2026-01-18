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
@pytest.mark.parametrize('how', ['start', 'end'])
def test_resample_to_quarterly_start_end(self, simple_period_range_series, how):
    ts = simple_period_range_series('1990', '1992', freq='Y-JUN')
    msg = "The 'convention' keyword in Series.resample is deprecated"
    with tm.assert_produces_warning(FutureWarning, match=msg):
        result = ts.resample('Q-MAR', convention=how).ffill()
    expected = ts.asfreq('Q-MAR', how=how)
    expected = expected.reindex(result.index, method='ffill')
    tm.assert_series_equal(result, expected)