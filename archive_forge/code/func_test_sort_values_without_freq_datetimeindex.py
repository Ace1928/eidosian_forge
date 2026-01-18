import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('index_dates,expected_dates', [(['2011-01-01', '2011-01-03', '2011-01-05', '2011-01-02', '2011-01-01'], ['2011-01-01', '2011-01-01', '2011-01-02', '2011-01-03', '2011-01-05']), (['2011-01-01', '2011-01-03', '2011-01-05', '2011-01-02', '2011-01-01'], ['2011-01-01', '2011-01-01', '2011-01-02', '2011-01-03', '2011-01-05']), ([NaT, '2011-01-03', '2011-01-05', '2011-01-02', NaT], [NaT, NaT, '2011-01-02', '2011-01-03', '2011-01-05'])])
def test_sort_values_without_freq_datetimeindex(self, index_dates, expected_dates, tz_naive_fixture):
    tz = tz_naive_fixture
    idx = DatetimeIndex(index_dates, tz=tz, name='idx')
    expected = DatetimeIndex(expected_dates, tz=tz, name='idx')
    self.check_sort_values_without_freq(idx, expected)