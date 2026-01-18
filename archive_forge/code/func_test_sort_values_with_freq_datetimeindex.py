import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('idx', [DatetimeIndex(['2011-01-01', '2011-01-02', '2011-01-03'], freq='D', name='idx'), DatetimeIndex(['2011-01-01 09:00', '2011-01-01 10:00', '2011-01-01 11:00'], freq='h', name='tzidx', tz='Asia/Tokyo')])
def test_sort_values_with_freq_datetimeindex(self, idx):
    self.check_sort_values_with_freq(idx)