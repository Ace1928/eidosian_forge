from datetime import (
import inspect
import numpy as np
import pytest
from pandas._libs.tslibs.timezones import dateutil_gettz as gettz
from pandas.compat import (
from pandas.compat.numpy import np_version_gt2
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.api.types import CategoricalDtype
@pytest.mark.xfail(not IS64 or (is_platform_windows() and (not np_version_gt2)), reason='Passes int32 values to DatetimeArray in make_na_array on windows, 32bit linux builds')
@td.skip_array_manager_not_yet_implemented
def test_reindex_tzaware_fill_value(self):
    df = DataFrame([[1]])
    ts = pd.Timestamp('2023-04-10 17:32', tz='US/Pacific')
    res = df.reindex([0, 1], axis=1, fill_value=ts)
    assert res.dtypes[1] == pd.DatetimeTZDtype(unit='s', tz='US/Pacific')
    expected = DataFrame({0: [1], 1: [ts]})
    expected[1] = expected[1].astype(res.dtypes[1])
    tm.assert_frame_equal(res, expected)
    per = ts.tz_localize(None).to_period('s')
    res = df.reindex([0, 1], axis=1, fill_value=per)
    assert res.dtypes[1] == pd.PeriodDtype('s')
    expected = DataFrame({0: [1], 1: [per]})
    tm.assert_frame_equal(res, expected)
    interval = pd.Interval(ts, ts + pd.Timedelta(seconds=1))
    res = df.reindex([0, 1], axis=1, fill_value=interval)
    assert res.dtypes[1] == pd.IntervalDtype('datetime64[s, US/Pacific]', 'right')
    expected = DataFrame({0: [1], 1: [interval]})
    expected[1] = expected[1].astype(res.dtypes[1])
    tm.assert_frame_equal(res, expected)