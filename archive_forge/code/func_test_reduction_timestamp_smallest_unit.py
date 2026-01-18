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
@td.skip_array_manager_invalid_test
def test_reduction_timestamp_smallest_unit(self):
    df = DataFrame({'a': Series([Timestamp('2019-12-31')], dtype='datetime64[s]'), 'b': Series([Timestamp('2019-12-31 00:00:00.123')], dtype='datetime64[ms]')})
    result = df.max()
    expected = Series([Timestamp('2019-12-31'), Timestamp('2019-12-31 00:00:00.123')], dtype='datetime64[ms]', index=['a', 'b'])
    tm.assert_series_equal(result, expected)