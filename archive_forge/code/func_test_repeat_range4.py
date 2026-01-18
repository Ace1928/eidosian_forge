import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_repeat_range4(self, tz_naive_fixture, unit):
    tz = tz_naive_fixture
    index = DatetimeIndex(['2001-01-01', 'NaT', '2003-01-01'], tz=tz).as_unit(unit)
    exp = DatetimeIndex(['2001-01-01', '2001-01-01', '2001-01-01', 'NaT', 'NaT', 'NaT', '2003-01-01', '2003-01-01', '2003-01-01'], tz=tz).as_unit(unit)
    for res in [index.repeat(3), np.repeat(index, 3)]:
        tm.assert_index_equal(res, exp)
        assert res.freq is None