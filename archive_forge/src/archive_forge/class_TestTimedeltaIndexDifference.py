import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tseries.offsets import Hour
class TestTimedeltaIndexDifference:

    def test_difference_freq(self, sort):
        index = timedelta_range('0 days', '5 days', freq='D')
        other = timedelta_range('1 days', '4 days', freq='D')
        expected = TimedeltaIndex(['0 days', '5 days'], freq=None)
        idx_diff = index.difference(other, sort)
        tm.assert_index_equal(idx_diff, expected)
        tm.assert_attr_equal('freq', idx_diff, expected)
        other = timedelta_range('2 days', '5 days', freq='D')
        idx_diff = index.difference(other, sort)
        expected = TimedeltaIndex(['0 days', '1 days'], freq='D')
        tm.assert_index_equal(idx_diff, expected)
        tm.assert_attr_equal('freq', idx_diff, expected)

    def test_difference_sort(self, sort):
        index = TimedeltaIndex(['5 days', '3 days', '2 days', '4 days', '1 days', '0 days'])
        other = timedelta_range('1 days', '4 days', freq='D')
        idx_diff = index.difference(other, sort)
        expected = TimedeltaIndex(['5 days', '0 days'], freq=None)
        if sort is None:
            expected = expected.sort_values()
        tm.assert_index_equal(idx_diff, expected)
        tm.assert_attr_equal('freq', idx_diff, expected)
        other = timedelta_range('2 days', '5 days', freq='D')
        idx_diff = index.difference(other, sort)
        expected = TimedeltaIndex(['1 days', '0 days'], freq=None)
        if sort is None:
            expected = expected.sort_values()
        tm.assert_index_equal(idx_diff, expected)
        tm.assert_attr_equal('freq', idx_diff, expected)