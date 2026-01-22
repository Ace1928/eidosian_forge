from datetime import (
import operator
import numpy as np
import pytest
from pandas.errors import OutOfBoundsTimedelta
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core import ops
class TestTimedeltaComparison:

    @pytest.mark.skip_ubsan
    def test_compare_pytimedelta_bounds(self):
        for unit in ['ns', 'us']:
            tdmax = Timedelta.max.as_unit(unit).max
            tdmin = Timedelta.min.as_unit(unit).min
            assert tdmax < timedelta.max
            assert tdmax <= timedelta.max
            assert not tdmax > timedelta.max
            assert not tdmax >= timedelta.max
            assert tdmax != timedelta.max
            assert not tdmax == timedelta.max
            assert tdmin > timedelta.min
            assert tdmin >= timedelta.min
            assert not tdmin < timedelta.min
            assert not tdmin <= timedelta.min
            assert tdmin != timedelta.min
            assert not tdmin == timedelta.min
        for unit in ['ms', 's']:
            tdmax = Timedelta.max.as_unit(unit).max
            tdmin = Timedelta.min.as_unit(unit).min
            assert tdmax > timedelta.max
            assert tdmax >= timedelta.max
            assert not tdmax < timedelta.max
            assert not tdmax <= timedelta.max
            assert tdmax != timedelta.max
            assert not tdmax == timedelta.max
            assert tdmin < timedelta.min
            assert tdmin <= timedelta.min
            assert not tdmin > timedelta.min
            assert not tdmin >= timedelta.min
            assert tdmin != timedelta.min
            assert not tdmin == timedelta.min

    def test_compare_pytimedelta_bounds2(self):
        pytd = timedelta(days=999999999, seconds=86399)
        td64 = np.timedelta64(pytd.days, 'D') + np.timedelta64(pytd.seconds, 's')
        td = Timedelta(td64)
        assert td.days == pytd.days
        assert td.seconds == pytd.seconds
        assert td == pytd
        assert not td != pytd
        assert not td < pytd
        assert not td > pytd
        assert td <= pytd
        assert td >= pytd
        td2 = td - Timedelta(seconds=1).as_unit('s')
        assert td2 != pytd
        assert not td2 == pytd
        assert td2 < pytd
        assert td2 <= pytd
        assert not td2 > pytd
        assert not td2 >= pytd

    def test_compare_tick(self, tick_classes):
        cls = tick_classes
        off = cls(4)
        td = off._as_pd_timedelta
        assert isinstance(td, Timedelta)
        assert td == off
        assert not td != off
        assert td <= off
        assert td >= off
        assert not td < off
        assert not td > off
        assert not td == 2 * off
        assert td != 2 * off
        assert td <= 2 * off
        assert td < 2 * off
        assert not td >= 2 * off
        assert not td > 2 * off

    def test_comparison_object_array(self):
        td = Timedelta('2 days')
        other = Timedelta('3 hours')
        arr = np.array([other, td], dtype=object)
        res = arr == td
        expected = np.array([False, True], dtype=bool)
        assert (res == expected).all()
        arr = np.array([[other, td], [td, other]], dtype=object)
        res = arr != td
        expected = np.array([[True, False], [False, True]], dtype=bool)
        assert res.shape == expected.shape
        assert (res == expected).all()

    def test_compare_timedelta_ndarray(self):
        periods = [Timedelta('0 days 01:00:00'), Timedelta('0 days 01:00:00')]
        arr = np.array(periods)
        result = arr[0] > arr
        expected = np.array([False, False])
        tm.assert_numpy_array_equal(result, expected)

    def test_compare_td64_ndarray(self):
        arr = np.arange(5).astype('timedelta64[ns]')
        td = Timedelta(arr[1])
        expected = np.array([False, True, False, False, False], dtype=bool)
        result = td == arr
        tm.assert_numpy_array_equal(result, expected)
        result = arr == td
        tm.assert_numpy_array_equal(result, expected)
        result = td != arr
        tm.assert_numpy_array_equal(result, ~expected)
        result = arr != td
        tm.assert_numpy_array_equal(result, ~expected)

    def test_compare_custom_object(self):
        """
        Make sure non supported operations on Timedelta returns NonImplemented
        and yields to other operand (GH#20829).
        """

        class CustomClass:

            def __init__(self, cmp_result=None) -> None:
                self.cmp_result = cmp_result

            def generic_result(self):
                if self.cmp_result is None:
                    return NotImplemented
                else:
                    return self.cmp_result

            def __eq__(self, other):
                return self.generic_result()

            def __gt__(self, other):
                return self.generic_result()
        t = Timedelta('1s')
        assert t != 'string'
        assert t != 1
        assert t != CustomClass()
        assert t != CustomClass(cmp_result=False)
        assert t < CustomClass(cmp_result=True)
        assert not t < CustomClass(cmp_result=False)
        assert t == CustomClass(cmp_result=True)

    @pytest.mark.parametrize('val', ['string', 1])
    def test_compare_unknown_type(self, val):
        t = Timedelta('1s')
        msg = "not supported between instances of 'Timedelta' and '(int|str)'"
        with pytest.raises(TypeError, match=msg):
            t >= val
        with pytest.raises(TypeError, match=msg):
            t > val
        with pytest.raises(TypeError, match=msg):
            t <= val
        with pytest.raises(TypeError, match=msg):
            t < val