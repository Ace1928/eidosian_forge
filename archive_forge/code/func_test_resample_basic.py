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
@pytest.mark.parametrize('_index_start,_index_end,_index_name', [('1/1/2000 00:00:00', '1/1/2000 00:13:00', 'index')])
@pytest.mark.parametrize('closed, expected', [('right', lambda s: Series([s.iloc[0], s[1:6].mean(), s[6:11].mean(), s[11:].mean()], index=date_range('1/1/2000', periods=4, freq='5min', name='index'))), ('left', lambda s: Series([s[:5].mean(), s[5:10].mean(), s[10:].mean()], index=date_range('1/1/2000 00:05', periods=3, freq='5min', name='index')))])
def test_resample_basic(series, closed, expected, unit):
    s = series
    s.index = s.index.as_unit(unit)
    expected = expected(s)
    expected.index = expected.index.as_unit(unit)
    result = s.resample('5min', closed=closed, label='right').mean()
    tm.assert_series_equal(result, expected)