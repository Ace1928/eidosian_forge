from datetime import (
import numpy as np
import pytest
from pandas._libs import index as libindex
from pandas.compat.numpy import np_long
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tseries.frequencies import to_offset
@pytest.mark.parametrize('tz', [None, 'US/Eastern', 'Asia/Tokyo'])
def test_take2(self, tz):
    dates = [datetime(2010, 1, 1, 14), datetime(2010, 1, 1, 15), datetime(2010, 1, 1, 17), datetime(2010, 1, 1, 21)]
    idx = date_range(start='2010-01-01 09:00', end='2010-02-01 09:00', freq='h', tz=tz, name='idx')
    expected = DatetimeIndex(dates, freq=None, name='idx', dtype=idx.dtype)
    taken1 = idx.take([5, 6, 8, 12])
    taken2 = idx[[5, 6, 8, 12]]
    for taken in [taken1, taken2]:
        tm.assert_index_equal(taken, expected)
        assert isinstance(taken, DatetimeIndex)
        assert taken.freq is None
        assert taken.tz == expected.tz
        assert taken.name == expected.name