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
def test_resample_how_method(unit):
    s = Series([11, 22], index=[Timestamp('2015-03-31 21:48:52.672000'), Timestamp('2015-03-31 21:49:52.739000')])
    s.index = s.index.as_unit(unit)
    expected = Series([11, np.nan, np.nan, np.nan, np.nan, np.nan, 22], index=DatetimeIndex([Timestamp('2015-03-31 21:48:50'), Timestamp('2015-03-31 21:49:00'), Timestamp('2015-03-31 21:49:10'), Timestamp('2015-03-31 21:49:20'), Timestamp('2015-03-31 21:49:30'), Timestamp('2015-03-31 21:49:40'), Timestamp('2015-03-31 21:49:50')], freq='10s'))
    expected.index = expected.index.as_unit(unit)
    tm.assert_series_equal(s.resample('10s').mean(), expected)