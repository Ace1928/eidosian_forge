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
@td.skip_array_manager_not_yet_implemented
def test_reduction_timedelta_smallest_unit(self):
    df = DataFrame({'a': Series([pd.Timedelta('1 days')], dtype='timedelta64[s]'), 'b': Series([pd.Timedelta('1 days')], dtype='timedelta64[ms]')})
    result = df.max()
    expected = Series([pd.Timedelta('1 days'), pd.Timedelta('1 days')], dtype='timedelta64[ms]', index=['a', 'b'])
    tm.assert_series_equal(result, expected)