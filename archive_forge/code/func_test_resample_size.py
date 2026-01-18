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
def test_resample_size(unit):
    n = 10000
    dr = date_range('2015-09-19', periods=n, freq='min').as_unit(unit)
    ts = Series(np.random.default_rng(2).standard_normal(n), index=np.random.default_rng(2).choice(dr, n))
    left = ts.resample('7min').size()
    ix = date_range(start=left.index.min(), end=ts.index.max(), freq='7min').as_unit(unit)
    bins = np.searchsorted(ix.values, ts.index.values, side='right')
    val = np.bincount(bins, minlength=len(ix) + 1)[1:].astype('int64', copy=False)
    right = Series(val, index=ix)
    tm.assert_series_equal(left, right)