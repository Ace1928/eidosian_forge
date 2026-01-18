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
def test_resample_segfault(unit):
    all_wins_and_wagers = [(1, datetime(2013, 10, 1, 16, 20), 1, 0), (2, datetime(2013, 10, 1, 16, 10), 1, 0), (2, datetime(2013, 10, 1, 18, 15), 1, 0), (2, datetime(2013, 10, 1, 16, 10, 31), 1, 0)]
    df = DataFrame.from_records(all_wins_and_wagers, columns=('ID', 'timestamp', 'A', 'B')).set_index('timestamp')
    df.index = df.index.as_unit(unit)
    msg = 'DataFrameGroupBy.resample operated on the grouping columns'
    with tm.assert_produces_warning(DeprecationWarning, match=msg):
        result = df.groupby('ID').resample('5min').sum()
    msg = 'DataFrameGroupBy.apply operated on the grouping columns'
    with tm.assert_produces_warning(DeprecationWarning, match=msg):
        expected = df.groupby('ID').apply(lambda x: x.resample('5min').sum())
    tm.assert_frame_equal(result, expected)