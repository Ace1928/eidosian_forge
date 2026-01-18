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
def test_resample_nunique(unit):
    df = DataFrame({'ID': {Timestamp('2015-06-05 00:00:00'): '0010100903', Timestamp('2015-06-08 00:00:00'): '0010150847'}, 'DATE': {Timestamp('2015-06-05 00:00:00'): '2015-06-05', Timestamp('2015-06-08 00:00:00'): '2015-06-08'}})
    df.index = df.index.as_unit(unit)
    r = df.resample('D')
    g = df.groupby(Grouper(freq='D'))
    expected = df.groupby(Grouper(freq='D')).ID.apply(lambda x: x.nunique())
    assert expected.name == 'ID'
    for t in [r, g]:
        result = t.ID.nunique()
        tm.assert_series_equal(result, expected)
    result = df.ID.resample('D').nunique()
    tm.assert_series_equal(result, expected)
    result = df.ID.groupby(Grouper(freq='D')).nunique()
    tm.assert_series_equal(result, expected)