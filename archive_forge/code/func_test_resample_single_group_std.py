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
def test_resample_single_group_std(unit):
    s = Series([30.1, 31.6], index=[Timestamp('20070915 15:30:00'), Timestamp('20070915 15:40:00')])
    s.index = s.index.as_unit(unit)
    expected = Series([0.75], index=DatetimeIndex([Timestamp('20070915')], freq='D').as_unit(unit))
    result = s.resample('D').apply(lambda x: np.std(x))
    tm.assert_series_equal(result, expected)