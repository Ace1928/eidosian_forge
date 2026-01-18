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
@pytest.mark.parametrize('label, sec', [[None, 2.0], ['right', '4.2']])
def test_resample_anchored_multiday(label, sec):
    index1 = date_range('2014-10-14 23:06:23.206', periods=3, freq='400ms')
    index2 = date_range('2014-10-15 23:00:00', periods=2, freq='2200ms')
    index = index1.union(index2)
    s = Series(np.random.default_rng(2).standard_normal(5), index=index)
    result = s.resample('2200ms', label=label).mean()
    assert result.index[-1] == Timestamp(f'2014-10-15 23:00:{sec}00')