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
def test_resample_consistency(unit):
    i30 = date_range('2002-02-02', periods=4, freq='30min').as_unit(unit)
    s = Series(np.arange(4.0), index=i30)
    s.iloc[2] = np.nan
    i10 = date_range(i30[0], i30[-1], freq='10min').as_unit(unit)
    s10 = s.reindex(index=i10, method='bfill')
    s10_2 = s.reindex(index=i10, method='bfill', limit=2)
    rl = s.reindex_like(s10, method='bfill', limit=2)
    r10_2 = s.resample('10Min').bfill(limit=2)
    r10 = s.resample('10Min').bfill()
    tm.assert_series_equal(s10_2, r10)
    tm.assert_series_equal(s10_2, r10_2)
    tm.assert_series_equal(s10_2, rl)