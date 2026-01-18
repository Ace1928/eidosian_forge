from datetime import (
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_set_index_datetime(self):
    df = DataFrame({'label': ['a', 'a', 'a', 'b', 'b', 'b'], 'datetime': ['2011-07-19 07:00:00', '2011-07-19 08:00:00', '2011-07-19 09:00:00', '2011-07-19 07:00:00', '2011-07-19 08:00:00', '2011-07-19 09:00:00'], 'value': range(6)})
    df.index = to_datetime(df.pop('datetime'), utc=True)
    df.index = df.index.tz_convert('US/Pacific')
    expected = DatetimeIndex(['2011-07-19 07:00:00', '2011-07-19 08:00:00', '2011-07-19 09:00:00'], name='datetime')
    expected = expected.tz_localize('UTC').tz_convert('US/Pacific')
    df = df.set_index('label', append=True)
    tm.assert_index_equal(df.index.levels[0], expected)
    tm.assert_index_equal(df.index.levels[1], Index(['a', 'b'], name='label'))
    assert df.index.names == ['datetime', 'label']
    df = df.swaplevel(0, 1)
    tm.assert_index_equal(df.index.levels[0], Index(['a', 'b'], name='label'))
    tm.assert_index_equal(df.index.levels[1], expected)
    assert df.index.names == ['label', 'datetime']
    df = DataFrame(np.random.default_rng(2).random(6))
    idx1 = DatetimeIndex(['2011-07-19 07:00:00', '2011-07-19 08:00:00', '2011-07-19 09:00:00', '2011-07-19 07:00:00', '2011-07-19 08:00:00', '2011-07-19 09:00:00'], tz='US/Eastern')
    idx2 = DatetimeIndex(['2012-04-01 09:00', '2012-04-01 09:00', '2012-04-01 09:00', '2012-04-02 09:00', '2012-04-02 09:00', '2012-04-02 09:00'], tz='US/Eastern')
    idx3 = date_range('2011-01-01 09:00', periods=6, tz='Asia/Tokyo')
    idx3 = idx3._with_freq(None)
    df = df.set_index(idx1)
    df = df.set_index(idx2, append=True)
    df = df.set_index(idx3, append=True)
    expected1 = DatetimeIndex(['2011-07-19 07:00:00', '2011-07-19 08:00:00', '2011-07-19 09:00:00'], tz='US/Eastern')
    expected2 = DatetimeIndex(['2012-04-01 09:00', '2012-04-02 09:00'], tz='US/Eastern')
    tm.assert_index_equal(df.index.levels[0], expected1)
    tm.assert_index_equal(df.index.levels[1], expected2)
    tm.assert_index_equal(df.index.levels[2], idx3)
    tm.assert_index_equal(df.index.get_level_values(0), idx1)
    tm.assert_index_equal(df.index.get_level_values(1), idx2)
    tm.assert_index_equal(df.index.get_level_values(2), idx3)