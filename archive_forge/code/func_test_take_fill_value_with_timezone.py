from datetime import (
import numpy as np
import pytest
from pandas._libs import index as libindex
from pandas.compat.numpy import np_long
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tseries.frequencies import to_offset
def test_take_fill_value_with_timezone(self):
    idx = DatetimeIndex(['2011-01-01', '2011-02-01', '2011-03-01'], name='xxx', tz='US/Eastern')
    result = idx.take(np.array([1, 0, -1]))
    expected = DatetimeIndex(['2011-02-01', '2011-01-01', '2011-03-01'], name='xxx', tz='US/Eastern')
    tm.assert_index_equal(result, expected)
    result = idx.take(np.array([1, 0, -1]), fill_value=True)
    expected = DatetimeIndex(['2011-02-01', '2011-01-01', 'NaT'], name='xxx', tz='US/Eastern')
    tm.assert_index_equal(result, expected)
    result = idx.take(np.array([1, 0, -1]), allow_fill=False, fill_value=True)
    expected = DatetimeIndex(['2011-02-01', '2011-01-01', '2011-03-01'], name='xxx', tz='US/Eastern')
    tm.assert_index_equal(result, expected)
    msg = 'When allow_fill=True and fill_value is not None, all indices must be >= -1'
    with pytest.raises(ValueError, match=msg):
        idx.take(np.array([1, 0, -2]), fill_value=True)
    with pytest.raises(ValueError, match=msg):
        idx.take(np.array([1, 0, -5]), fill_value=True)
    msg = 'out of bounds'
    with pytest.raises(IndexError, match=msg):
        idx.take(np.array([1, -5]))