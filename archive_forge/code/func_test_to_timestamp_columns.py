from datetime import timedelta
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_to_timestamp_columns(self):
    K = 5
    index = period_range(freq='Y', start='1/1/2001', end='12/1/2009')
    df = DataFrame(np.random.default_rng(2).standard_normal((len(index), K)), index=index, columns=['A', 'B', 'C', 'D', 'E'])
    df['mix'] = 'a'
    df = df.T
    exp_index = date_range('1/1/2001', end='12/31/2009', freq='YE-DEC')
    exp_index = exp_index + Timedelta(1, 'D') - Timedelta(1, 'ns')
    result = df.to_timestamp('D', 'end', axis=1)
    tm.assert_index_equal(result.columns, exp_index)
    tm.assert_numpy_array_equal(result.values, df.values)
    exp_index = date_range('1/1/2001', end='1/1/2009', freq='YS-JAN')
    result = df.to_timestamp('D', 'start', axis=1)
    tm.assert_index_equal(result.columns, exp_index)
    delta = timedelta(hours=23)
    result = df.to_timestamp('H', 'end', axis=1)
    exp_index = _get_with_delta(delta)
    exp_index = exp_index + Timedelta(1, 'h') - Timedelta(1, 'ns')
    tm.assert_index_equal(result.columns, exp_index)
    delta = timedelta(hours=23, minutes=59)
    result = df.to_timestamp('min', 'end', axis=1)
    exp_index = _get_with_delta(delta)
    exp_index = exp_index + Timedelta(1, 'm') - Timedelta(1, 'ns')
    tm.assert_index_equal(result.columns, exp_index)
    result = df.to_timestamp('S', 'end', axis=1)
    delta = timedelta(hours=23, minutes=59, seconds=59)
    exp_index = _get_with_delta(delta)
    exp_index = exp_index + Timedelta(1, 's') - Timedelta(1, 'ns')
    tm.assert_index_equal(result.columns, exp_index)
    result1 = df.to_timestamp('5min', axis=1)
    result2 = df.to_timestamp('min', axis=1)
    expected = date_range('2001-01-01', '2009-01-01', freq='YS')
    assert isinstance(result1.columns, DatetimeIndex)
    assert isinstance(result2.columns, DatetimeIndex)
    tm.assert_numpy_array_equal(result1.columns.asi8, expected.asi8)
    tm.assert_numpy_array_equal(result2.columns.asi8, expected.asi8)
    assert result1.columns.freqstr == 'YS-JAN'
    assert result2.columns.freqstr == 'YS-JAN'