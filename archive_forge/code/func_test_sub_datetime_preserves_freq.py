import pytest
from pandas import (
import pandas._testing as tm
def test_sub_datetime_preserves_freq(self, tz_naive_fixture):
    dti = date_range('2016-01-01', periods=12, tz=tz_naive_fixture)
    res = dti - dti[0]
    expected = timedelta_range('0 Days', '11 Days')
    tm.assert_index_equal(res, expected)
    assert res.freq == expected.freq