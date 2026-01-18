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
def test_resample_how(series, downsample_method, unit):
    if downsample_method == 'ohlc':
        pytest.skip('covered by test_resample_how_ohlc')
    s = series
    s.index = s.index.as_unit(unit)
    grouplist = np.ones_like(s)
    grouplist[0] = 0
    grouplist[1:6] = 1
    grouplist[6:11] = 2
    grouplist[11:] = 3
    expected = s.groupby(grouplist).agg(downsample_method)
    expected.index = date_range('1/1/2000', periods=4, freq='5min', name='index').as_unit(unit)
    result = getattr(s.resample('5min', closed='right', label='right'), downsample_method)()
    tm.assert_series_equal(result, expected)