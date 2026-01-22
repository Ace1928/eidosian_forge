from datetime import (
import operator
import numpy as np
import pytest
from pandas import Timestamp
import pandas._testing as tm
class TestTimestampComparison:

    def test_compare_non_nano_dt64(self):
        dt = np.datetime64('1066-10-14')
        ts = Timestamp(dt)
        assert dt == ts

    def test_comparison_dt64_ndarray(self):
        ts = Timestamp('2021-01-01')
        ts2 = Timestamp('2019-04-05')
        arr = np.array([[ts.asm8, ts2.asm8]], dtype='M8[ns]')
        result = ts == arr
        expected = np.array([[True, False]], dtype=bool)
        tm.assert_numpy_array_equal(result, expected)
        result = arr == ts
        tm.assert_numpy_array_equal(result, expected)
        result = ts != arr
        tm.assert_numpy_array_equal(result, ~expected)
        result = arr != ts
        tm.assert_numpy_array_equal(result, ~expected)
        result = ts2 < arr
        tm.assert_numpy_array_equal(result, expected)
        result = arr < ts2
        tm.assert_numpy_array_equal(result, np.array([[False, False]], dtype=bool))
        result = ts2 <= arr
        tm.assert_numpy_array_equal(result, np.array([[True, True]], dtype=bool))
        result = arr <= ts2
        tm.assert_numpy_array_equal(result, ~expected)
        result = ts >= arr
        tm.assert_numpy_array_equal(result, np.array([[True, True]], dtype=bool))
        result = arr >= ts
        tm.assert_numpy_array_equal(result, np.array([[True, False]], dtype=bool))

    @pytest.mark.parametrize('reverse', [True, False])
    def test_comparison_dt64_ndarray_tzaware(self, reverse, comparison_op):
        ts = Timestamp('2021-01-01 00:00:00.00000', tz='UTC')
        arr = np.array([ts.asm8, ts.asm8], dtype='M8[ns]')
        left, right = (ts, arr)
        if reverse:
            left, right = (arr, ts)
        if comparison_op is operator.eq:
            expected = np.array([False, False], dtype=bool)
            result = comparison_op(left, right)
            tm.assert_numpy_array_equal(result, expected)
        elif comparison_op is operator.ne:
            expected = np.array([True, True], dtype=bool)
            result = comparison_op(left, right)
            tm.assert_numpy_array_equal(result, expected)
        else:
            msg = 'Cannot compare tz-naive and tz-aware timestamps'
            with pytest.raises(TypeError, match=msg):
                comparison_op(left, right)

    def test_comparison_object_array(self):
        ts = Timestamp('2011-01-03 00:00:00-0500', tz='US/Eastern')
        other = Timestamp('2011-01-01 00:00:00-0500', tz='US/Eastern')
        naive = Timestamp('2011-01-01 00:00:00')
        arr = np.array([other, ts], dtype=object)
        res = arr == ts
        expected = np.array([False, True], dtype=bool)
        assert (res == expected).all()
        arr = np.array([[other, ts], [ts, other]], dtype=object)
        res = arr != ts
        expected = np.array([[True, False], [False, True]], dtype=bool)
        assert res.shape == expected.shape
        assert (res == expected).all()
        arr = np.array([naive], dtype=object)
        msg = 'Cannot compare tz-naive and tz-aware timestamps'
        with pytest.raises(TypeError, match=msg):
            arr < ts

    def test_comparison(self):
        stamp = 1337299200000000000
        val = Timestamp(stamp)
        assert val == val
        assert not val != val
        assert not val < val
        assert val <= val
        assert not val > val
        assert val >= val
        other = datetime(2012, 5, 18)
        assert val == other
        assert not val != other
        assert not val < other
        assert val <= other
        assert not val > other
        assert val >= other
        other = Timestamp(stamp + 100)
        assert val != other
        assert val != other
        assert val < other
        assert val <= other
        assert other > val
        assert other >= val

    def test_compare_invalid(self):
        val = Timestamp('20130101 12:01:02')
        assert not val == 'foo'
        assert not val == 10.0
        assert not val == 1
        assert not val == []
        assert not val == {'foo': 1}
        assert not val == np.float64(1)
        assert not val == np.int64(1)
        assert val != 'foo'
        assert val != 10.0
        assert val != 1
        assert val != []
        assert val != {'foo': 1}
        assert val != np.float64(1)
        assert val != np.int64(1)

    @pytest.mark.parametrize('tz', [None, 'US/Pacific'])
    def test_compare_date(self, tz):
        ts = Timestamp('2021-01-01 00:00:00.00000', tz=tz)
        dt = ts.to_pydatetime().date()
        msg = 'Cannot compare Timestamp with datetime.date'
        for left, right in [(ts, dt), (dt, ts)]:
            assert not left == right
            assert left != right
            with pytest.raises(TypeError, match=msg):
                left < right
            with pytest.raises(TypeError, match=msg):
                left <= right
            with pytest.raises(TypeError, match=msg):
                left > right
            with pytest.raises(TypeError, match=msg):
                left >= right

    def test_cant_compare_tz_naive_w_aware(self, utc_fixture):
        a = Timestamp('3/12/2012')
        b = Timestamp('3/12/2012', tz=utc_fixture)
        msg = 'Cannot compare tz-naive and tz-aware timestamps'
        assert not a == b
        assert a != b
        with pytest.raises(TypeError, match=msg):
            a < b
        with pytest.raises(TypeError, match=msg):
            a <= b
        with pytest.raises(TypeError, match=msg):
            a > b
        with pytest.raises(TypeError, match=msg):
            a >= b
        assert not b == a
        assert b != a
        with pytest.raises(TypeError, match=msg):
            b < a
        with pytest.raises(TypeError, match=msg):
            b <= a
        with pytest.raises(TypeError, match=msg):
            b > a
        with pytest.raises(TypeError, match=msg):
            b >= a
        assert not a == b.to_pydatetime()
        assert not a.to_pydatetime() == b

    def test_timestamp_compare_scalars(self):
        lhs = np.datetime64(datetime(2013, 12, 6))
        rhs = Timestamp('now')
        nat = Timestamp('nat')
        ops = {'gt': 'lt', 'lt': 'gt', 'ge': 'le', 'le': 'ge', 'eq': 'eq', 'ne': 'ne'}
        for left, right in ops.items():
            left_f = getattr(operator, left)
            right_f = getattr(operator, right)
            expected = left_f(lhs, rhs)
            result = right_f(rhs, lhs)
            assert result == expected
            expected = left_f(rhs, nat)
            result = right_f(nat, rhs)
            assert result == expected

    def test_timestamp_compare_with_early_datetime(self):
        stamp = Timestamp('2012-01-01')
        assert not stamp == datetime.min
        assert not stamp == datetime(1600, 1, 1)
        assert not stamp == datetime(2700, 1, 1)
        assert stamp != datetime.min
        assert stamp != datetime(1600, 1, 1)
        assert stamp != datetime(2700, 1, 1)
        assert stamp > datetime(1600, 1, 1)
        assert stamp >= datetime(1600, 1, 1)
        assert stamp < datetime(2700, 1, 1)
        assert stamp <= datetime(2700, 1, 1)
        other = Timestamp.min.to_pydatetime(warn=False)
        assert other - timedelta(microseconds=1) < Timestamp.min

    def test_timestamp_compare_oob_dt64(self):
        us = np.timedelta64(1, 'us')
        other = np.datetime64(Timestamp.min).astype('M8[us]')
        assert Timestamp.min > other
        other = np.datetime64(Timestamp.max).astype('M8[us]')
        assert Timestamp.max > other
        assert other < Timestamp.max
        assert Timestamp.max < other + us
        other = datetime(9999, 9, 9)
        assert Timestamp.min < other
        assert other > Timestamp.min
        assert Timestamp.max < other
        assert other > Timestamp.max
        other = datetime(1, 1, 1)
        assert Timestamp.max > other
        assert other < Timestamp.max
        assert Timestamp.min > other
        assert other < Timestamp.min

    def test_compare_zerodim_array(self, fixed_now_ts):
        ts = fixed_now_ts
        dt64 = np.datetime64('2016-01-01', 'ns')
        arr = np.array(dt64)
        assert arr.ndim == 0
        result = arr < ts
        assert result is np.bool_(True)
        result = arr > ts
        assert result is np.bool_(False)