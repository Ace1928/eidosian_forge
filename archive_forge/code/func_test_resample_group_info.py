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
@pytest.mark.parametrize('n', [10000, 100000])
@pytest.mark.parametrize('k', [10, 100, 1000])
def test_resample_group_info(n, k, unit):
    prng = np.random.default_rng(2)
    dr = date_range(start='2015-08-27', periods=n // 10, freq='min').as_unit(unit)
    ts = Series(prng.integers(0, n // k, n).astype('int64'), index=prng.choice(dr, n))
    left = ts.resample('30min').nunique()
    ix = date_range(start=ts.index.min(), end=ts.index.max(), freq='30min').as_unit(unit)
    vals = ts.values
    bins = np.searchsorted(ix.values, ts.index, side='right')
    sorter = np.lexsort((vals, bins))
    vals, bins = (vals[sorter], bins[sorter])
    mask = np.r_[True, vals[1:] != vals[:-1]]
    mask |= np.r_[True, bins[1:] != bins[:-1]]
    arr = np.bincount(bins[mask] - 1, minlength=len(ix)).astype('int64', copy=False)
    right = Series(arr, index=ix)
    tm.assert_series_equal(left, right)