import pytest
from pandas._libs.tslibs import to_offset
from pandas._libs.tslibs.offsets import INVALID_FREQ_ERR_MSG
from pandas import (
import pandas._testing as tm
def test_round3(self, tz_naive_fixture):
    tz = tz_naive_fixture
    index = DatetimeIndex(['2016-10-17 12:00:00.00149'], tz=tz).as_unit('ns')
    result = index.round('ms')
    expected = DatetimeIndex(['2016-10-17 12:00:00.001000'], tz=tz).as_unit('ns')
    tm.assert_index_equal(result, expected)