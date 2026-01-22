from datetime import timedelta
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
class TestIntervalArithmetic:

    def test_interval_add(self, closed):
        interval = Interval(0, 1, closed=closed)
        expected = Interval(1, 2, closed=closed)
        result = interval + 1
        assert result == expected
        result = 1 + interval
        assert result == expected
        result = interval
        result += 1
        assert result == expected
        msg = 'unsupported operand type\\(s\\) for \\+'
        with pytest.raises(TypeError, match=msg):
            interval + interval
        with pytest.raises(TypeError, match=msg):
            interval + 'foo'

    def test_interval_sub(self, closed):
        interval = Interval(0, 1, closed=closed)
        expected = Interval(-1, 0, closed=closed)
        result = interval - 1
        assert result == expected
        result = interval
        result -= 1
        assert result == expected
        msg = 'unsupported operand type\\(s\\) for -'
        with pytest.raises(TypeError, match=msg):
            interval - interval
        with pytest.raises(TypeError, match=msg):
            interval - 'foo'

    def test_interval_mult(self, closed):
        interval = Interval(0, 1, closed=closed)
        expected = Interval(0, 2, closed=closed)
        result = interval * 2
        assert result == expected
        result = 2 * interval
        assert result == expected
        result = interval
        result *= 2
        assert result == expected
        msg = 'unsupported operand type\\(s\\) for \\*'
        with pytest.raises(TypeError, match=msg):
            interval * interval
        msg = "can\\'t multiply sequence by non-int"
        with pytest.raises(TypeError, match=msg):
            interval * 'foo'

    def test_interval_div(self, closed):
        interval = Interval(0, 1, closed=closed)
        expected = Interval(0, 0.5, closed=closed)
        result = interval / 2.0
        assert result == expected
        result = interval
        result /= 2.0
        assert result == expected
        msg = 'unsupported operand type\\(s\\) for /'
        with pytest.raises(TypeError, match=msg):
            interval / interval
        with pytest.raises(TypeError, match=msg):
            interval / 'foo'

    def test_interval_floordiv(self, closed):
        interval = Interval(1, 2, closed=closed)
        expected = Interval(0, 1, closed=closed)
        result = interval // 2
        assert result == expected
        result = interval
        result //= 2
        assert result == expected
        msg = 'unsupported operand type\\(s\\) for //'
        with pytest.raises(TypeError, match=msg):
            interval // interval
        with pytest.raises(TypeError, match=msg):
            interval // 'foo'

    @pytest.mark.parametrize('method', ['__add__', '__sub__'])
    @pytest.mark.parametrize('interval', [Interval(Timestamp('2017-01-01 00:00:00'), Timestamp('2018-01-01 00:00:00')), Interval(Timedelta(days=7), Timedelta(days=14))])
    @pytest.mark.parametrize('delta', [Timedelta(days=7), timedelta(7), np.timedelta64(7, 'D')])
    def test_time_interval_add_subtract_timedelta(self, interval, delta, method):
        result = getattr(interval, method)(delta)
        left = getattr(interval.left, method)(delta)
        right = getattr(interval.right, method)(delta)
        expected = Interval(left, right)
        assert result == expected

    @pytest.mark.parametrize('interval', [Interval(1, 2), Interval(1.0, 2.0)])
    @pytest.mark.parametrize('delta', [Timedelta(days=7), timedelta(7), np.timedelta64(7, 'D')])
    def test_numeric_interval_add_timedelta_raises(self, interval, delta):
        msg = '|'.join(['unsupported operand', 'cannot use operands', 'Only numeric, Timestamp and Timedelta endpoints are allowed'])
        with pytest.raises((TypeError, ValueError), match=msg):
            interval + delta
        with pytest.raises((TypeError, ValueError), match=msg):
            delta + interval

    @pytest.mark.parametrize('klass', [timedelta, np.timedelta64, Timedelta])
    def test_timedelta_add_timestamp_interval(self, klass):
        delta = klass(0)
        expected = Interval(Timestamp('2020-01-01'), Timestamp('2020-02-01'))
        result = delta + expected
        assert result == expected
        result = expected + delta
        assert result == expected