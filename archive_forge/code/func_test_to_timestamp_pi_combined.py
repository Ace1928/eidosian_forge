from datetime import datetime
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_to_timestamp_pi_combined(self):
    idx = period_range(start='2011', periods=2, freq='1D1h', name='idx')
    result = idx.to_timestamp()
    expected = DatetimeIndex(['2011-01-01 00:00', '2011-01-02 01:00'], dtype='M8[ns]', name='idx')
    tm.assert_index_equal(result, expected)
    result = idx.to_timestamp(how='E')
    expected = DatetimeIndex(['2011-01-02 00:59:59', '2011-01-03 01:59:59'], name='idx', dtype='M8[ns]')
    expected = expected + Timedelta(1, 's') - Timedelta(1, 'ns')
    tm.assert_index_equal(result, expected)
    result = idx.to_timestamp(how='E', freq='h')
    expected = DatetimeIndex(['2011-01-02 00:00', '2011-01-03 01:00'], dtype='M8[ns]', name='idx')
    expected = expected + Timedelta(1, 'h') - Timedelta(1, 'ns')
    tm.assert_index_equal(result, expected)