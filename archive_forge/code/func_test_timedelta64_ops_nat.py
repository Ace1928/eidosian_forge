from datetime import (
import numpy as np
import pytest
from pandas.errors import (
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import NumpyExtensionArray
from pandas.tests.arithmetic.common import (
def test_timedelta64_ops_nat(self):
    timedelta_series = Series([NaT, Timedelta('1s')])
    nat_series_dtype_timedelta = Series([NaT, NaT], dtype='timedelta64[ns]')
    single_nat_dtype_timedelta = Series([NaT], dtype='timedelta64[ns]')
    tm.assert_series_equal(timedelta_series - NaT, nat_series_dtype_timedelta)
    tm.assert_series_equal(-NaT + timedelta_series, nat_series_dtype_timedelta)
    tm.assert_series_equal(timedelta_series - single_nat_dtype_timedelta, nat_series_dtype_timedelta)
    tm.assert_series_equal(-single_nat_dtype_timedelta + timedelta_series, nat_series_dtype_timedelta)
    tm.assert_series_equal(nat_series_dtype_timedelta + NaT, nat_series_dtype_timedelta)
    tm.assert_series_equal(NaT + nat_series_dtype_timedelta, nat_series_dtype_timedelta)
    tm.assert_series_equal(nat_series_dtype_timedelta + single_nat_dtype_timedelta, nat_series_dtype_timedelta)
    tm.assert_series_equal(single_nat_dtype_timedelta + nat_series_dtype_timedelta, nat_series_dtype_timedelta)
    tm.assert_series_equal(timedelta_series + NaT, nat_series_dtype_timedelta)
    tm.assert_series_equal(NaT + timedelta_series, nat_series_dtype_timedelta)
    tm.assert_series_equal(timedelta_series + single_nat_dtype_timedelta, nat_series_dtype_timedelta)
    tm.assert_series_equal(single_nat_dtype_timedelta + timedelta_series, nat_series_dtype_timedelta)
    tm.assert_series_equal(nat_series_dtype_timedelta + NaT, nat_series_dtype_timedelta)
    tm.assert_series_equal(NaT + nat_series_dtype_timedelta, nat_series_dtype_timedelta)
    tm.assert_series_equal(nat_series_dtype_timedelta + single_nat_dtype_timedelta, nat_series_dtype_timedelta)
    tm.assert_series_equal(single_nat_dtype_timedelta + nat_series_dtype_timedelta, nat_series_dtype_timedelta)
    tm.assert_series_equal(nat_series_dtype_timedelta * 1.0, nat_series_dtype_timedelta)
    tm.assert_series_equal(1.0 * nat_series_dtype_timedelta, nat_series_dtype_timedelta)
    tm.assert_series_equal(timedelta_series * 1, timedelta_series)
    tm.assert_series_equal(1 * timedelta_series, timedelta_series)
    tm.assert_series_equal(timedelta_series * 1.5, Series([NaT, Timedelta('1.5s')]))
    tm.assert_series_equal(1.5 * timedelta_series, Series([NaT, Timedelta('1.5s')]))
    tm.assert_series_equal(timedelta_series * np.nan, nat_series_dtype_timedelta)
    tm.assert_series_equal(np.nan * timedelta_series, nat_series_dtype_timedelta)
    tm.assert_series_equal(timedelta_series / 2, Series([NaT, Timedelta('0.5s')]))
    tm.assert_series_equal(timedelta_series / 2.0, Series([NaT, Timedelta('0.5s')]))
    tm.assert_series_equal(timedelta_series / np.nan, nat_series_dtype_timedelta)