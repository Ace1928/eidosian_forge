from datetime import timedelta
from decimal import Decimal
import re
from dateutil.tz import tzlocal
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
from pandas.compat import (
from pandas.compat.numpy import np_version_gt2
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core import (
def test_std_timedelta64_skipna_false(self):
    tdi = pd.timedelta_range('1 Day', periods=10)
    df = DataFrame({'A': tdi, 'B': tdi}, copy=True)
    df.iloc[-2, -1] = pd.NaT
    result = df.std(skipna=False)
    expected = Series([df['A'].std(), pd.NaT], index=['A', 'B'], dtype='timedelta64[ns]')
    tm.assert_series_equal(result, expected)
    result = df.std(axis=1, skipna=False)
    expected = Series([pd.Timedelta(0)] * 8 + [pd.NaT, pd.Timedelta(0)])
    tm.assert_series_equal(result, expected)