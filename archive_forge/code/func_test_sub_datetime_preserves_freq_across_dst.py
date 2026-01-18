import pytest
from pandas import (
import pandas._testing as tm
@pytest.mark.xfail(reason='The inherited freq is incorrect bc dti.freq is incorrect https://github.com/pandas-dev/pandas/pull/48818/files#r982793461')
def test_sub_datetime_preserves_freq_across_dst(self):
    ts = Timestamp('2016-03-11', tz='US/Pacific')
    dti = date_range(ts, periods=4)
    res = dti - dti[0]
    expected = TimedeltaIndex([Timedelta(days=0), Timedelta(days=1), Timedelta(days=2), Timedelta(days=2, hours=23)])
    tm.assert_index_equal(res, expected)
    assert res.freq == expected.freq