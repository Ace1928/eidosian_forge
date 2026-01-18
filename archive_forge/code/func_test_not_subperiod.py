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
@pytest.mark.parametrize('rule,expected_error_msg', [('Y-DEC', '<YearEnd: month=12>'), ('Q-MAR', '<QuarterEnd: startingMonth=3>'), ('M', '<MonthEnd>'), ('w-thu', '<Week: weekday=3>')])
def test_not_subperiod(self, simple_period_range_series, rule, expected_error_msg):
    ts = simple_period_range_series('1/1/1990', '6/30/1995', freq='w-wed')
    msg = f'Frequency <Week: weekday=2> cannot be resampled to {expected_error_msg}, as they are not sub or super periods'
    with pytest.raises(IncompatibleFrequency, match=msg):
        ts.resample(rule).mean()