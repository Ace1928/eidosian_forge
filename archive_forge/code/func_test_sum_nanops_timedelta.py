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
def test_sum_nanops_timedelta(self):
    idx = ['a', 'b', 'c']
    df = DataFrame({'a': [0, 0], 'b': [0, np.nan], 'c': [np.nan, np.nan]})
    df2 = df.apply(to_timedelta)
    result = df2.sum()
    expected = Series([0, 0, 0], dtype='m8[ns]', index=idx)
    tm.assert_series_equal(result, expected)
    result = df2.sum(min_count=0)
    tm.assert_series_equal(result, expected)
    result = df2.sum(min_count=1)
    expected = Series([0, 0, np.nan], dtype='m8[ns]', index=idx)
    tm.assert_series_equal(result, expected)