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
@pytest.mark.parametrize('func', [lambda x: x.nunique(), lambda x: x.agg(Series.nunique), lambda x: x.agg('nunique')], ids=['nunique', 'series_nunique', 'nunique_str'])
def test_resample_nunique_with_date_gap(func, unit):
    index = date_range('1-1-2000', '2-15-2000', freq='h').as_unit(unit)
    index2 = date_range('4-15-2000', '5-15-2000', freq='h').as_unit(unit)
    index3 = index.append(index2)
    s = Series(range(len(index3)), index=index3, dtype='int64')
    r = s.resample('ME')
    result = r.count()
    expected = func(r)
    tm.assert_series_equal(result, expected)