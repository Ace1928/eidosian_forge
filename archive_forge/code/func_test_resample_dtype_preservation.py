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
def test_resample_dtype_preservation(unit):
    df = DataFrame({'date': date_range(start='2016-01-01', periods=4, freq='W').as_unit(unit), 'group': [1, 1, 2, 2], 'val': Series([5, 6, 7, 8], dtype='int32')}).set_index('date')
    result = df.resample('1D').ffill()
    assert result.val.dtype == np.int32
    msg = 'DataFrameGroupBy.resample operated on the grouping columns'
    with tm.assert_produces_warning(DeprecationWarning, match=msg):
        result = df.groupby('group').resample('1D').ffill()
    assert result.val.dtype == np.int32