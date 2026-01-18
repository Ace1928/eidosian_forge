from datetime import (
from itertools import (
import operator
import numpy as np
import pytest
import pytz
from pandas._libs.tslibs.conversion import localize_pydatetime
from pandas._libs.tslibs.offsets import shift_months
from pandas.errors import PerformanceWarning
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core import roperator
from pandas.tests.arithmetic.common import (
def test_ops_nat_mixed_datetime64_timedelta64(self):
    timedelta_series = Series([NaT, Timedelta('1s')])
    datetime_series = Series([NaT, Timestamp('19900315')])
    nat_series_dtype_timedelta = Series([NaT, NaT], dtype='timedelta64[ns]')
    nat_series_dtype_timestamp = Series([NaT, NaT], dtype='datetime64[ns]')
    single_nat_dtype_datetime = Series([NaT], dtype='datetime64[ns]')
    single_nat_dtype_timedelta = Series([NaT], dtype='timedelta64[ns]')
    tm.assert_series_equal(datetime_series - single_nat_dtype_datetime, nat_series_dtype_timedelta)
    tm.assert_series_equal(datetime_series - single_nat_dtype_timedelta, nat_series_dtype_timestamp)
    tm.assert_series_equal(-single_nat_dtype_timedelta + datetime_series, nat_series_dtype_timestamp)
    tm.assert_series_equal(nat_series_dtype_timestamp - single_nat_dtype_datetime, nat_series_dtype_timedelta)
    tm.assert_series_equal(nat_series_dtype_timestamp - single_nat_dtype_timedelta, nat_series_dtype_timestamp)
    tm.assert_series_equal(-single_nat_dtype_timedelta + nat_series_dtype_timestamp, nat_series_dtype_timestamp)
    msg = 'cannot subtract a datelike'
    with pytest.raises(TypeError, match=msg):
        timedelta_series - single_nat_dtype_datetime
    tm.assert_series_equal(nat_series_dtype_timestamp + single_nat_dtype_timedelta, nat_series_dtype_timestamp)
    tm.assert_series_equal(single_nat_dtype_timedelta + nat_series_dtype_timestamp, nat_series_dtype_timestamp)
    tm.assert_series_equal(nat_series_dtype_timestamp + single_nat_dtype_timedelta, nat_series_dtype_timestamp)
    tm.assert_series_equal(single_nat_dtype_timedelta + nat_series_dtype_timestamp, nat_series_dtype_timestamp)
    tm.assert_series_equal(nat_series_dtype_timedelta + single_nat_dtype_datetime, nat_series_dtype_timestamp)
    tm.assert_series_equal(single_nat_dtype_datetime + nat_series_dtype_timedelta, nat_series_dtype_timestamp)