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
@pytest.mark.parametrize('func', ['min', 'max', 'first', 'last'])
def test_resample_aggregate_functions_min_count(func, unit):
    index = date_range(start='2020', freq='ME', periods=3).as_unit(unit)
    ser = Series([1, np.nan, np.nan], index)
    result = getattr(ser.resample('QE'), func)(min_count=2)
    expected = Series([np.nan], index=DatetimeIndex(['2020-03-31'], freq='QE-DEC').as_unit(unit))
    tm.assert_series_equal(result, expected)