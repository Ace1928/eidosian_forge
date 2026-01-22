from datetime import (
import operator
import numpy as np
import pytest
from pandas.errors import OutOfBoundsTimedelta
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core import ops
class TestTimedeltaMultiplicationDivision:
    """
    Tests for Timedelta methods:

        __mul__, __rmul__,
        __div__, __rdiv__,
        __truediv__, __rtruediv__,
        __floordiv__, __rfloordiv__,
        __mod__, __rmod__,
        __divmod__, __rdivmod__
    """

    @pytest.mark.parametrize('td_nat', [NaT, np.timedelta64('NaT', 'ns'), np.timedelta64('NaT')])
    @pytest.mark.parametrize('op', [operator.mul, ops.rmul])
    def test_td_mul_nat(self, op, td_nat):
        td = Timedelta(10, unit='d')
        typs = '|'.join(['numpy.timedelta64', 'NaTType', 'Timedelta'])
        msg = '|'.join([f"unsupported operand type\\(s\\) for \\*: '{typs}' and '{typs}'", "ufunc '?multiply'? cannot use operands with types"])
        with pytest.raises(TypeError, match=msg):
            op(td, td_nat)

    @pytest.mark.parametrize('nan', [np.nan, np.float64('NaN'), float('nan')])
    @pytest.mark.parametrize('op', [operator.mul, ops.rmul])
    def test_td_mul_nan(self, op, nan):
        td = Timedelta(10, unit='d')
        result = op(td, nan)
        assert result is NaT

    @pytest.mark.parametrize('op', [operator.mul, ops.rmul])
    def test_td_mul_scalar(self, op):
        td = Timedelta(minutes=3)
        result = op(td, 2)
        assert result == Timedelta(minutes=6)
        result = op(td, 1.5)
        assert result == Timedelta(minutes=4, seconds=30)
        assert op(td, np.nan) is NaT
        assert op(-1, td)._value == -1 * td._value
        assert op(-1.0, td)._value == -1.0 * td._value
        msg = 'unsupported operand type'
        with pytest.raises(TypeError, match=msg):
            op(td, Timestamp(2016, 1, 2))
        with pytest.raises(TypeError, match=msg):
            op(td, td)

    def test_td_mul_numeric_ndarray(self):
        td = Timedelta('1 day')
        other = np.array([2])
        expected = np.array([Timedelta('2 Days').to_timedelta64()])
        result = td * other
        tm.assert_numpy_array_equal(result, expected)
        result = other * td
        tm.assert_numpy_array_equal(result, expected)

    def test_td_mul_numeric_ndarray_0d(self):
        td = Timedelta('1 day')
        other = np.array(2)
        assert other.ndim == 0
        expected = Timedelta('2 days')
        res = td * other
        assert type(res) is Timedelta
        assert res == expected
        res = other * td
        assert type(res) is Timedelta
        assert res == expected

    def test_td_mul_td64_ndarray_invalid(self):
        td = Timedelta('1 day')
        other = np.array([Timedelta('2 Days').to_timedelta64()])
        msg = f"ufunc '?multiply'? cannot use operands with types dtype\\('{tm.ENDIAN}m8\\[ns\\]'\\) and dtype\\('{tm.ENDIAN}m8\\[ns\\]'\\)"
        with pytest.raises(TypeError, match=msg):
            td * other
        with pytest.raises(TypeError, match=msg):
            other * td

    def test_td_div_timedeltalike_scalar(self):
        td = Timedelta(10, unit='d')
        result = td / offsets.Hour(1)
        assert result == 240
        assert td / td == 1
        assert td / np.timedelta64(60, 'h') == 4
        assert np.isnan(td / NaT)

    def test_td_div_td64_non_nano(self):
        td = Timedelta('1 days 2 hours 3 ns')
        result = td / np.timedelta64(1, 'D')
        assert result == td._value / (86400 * 10 ** 9)
        result = td / np.timedelta64(1, 's')
        assert result == td._value / 10 ** 9
        result = td / np.timedelta64(1, 'ns')
        assert result == td._value
        td = Timedelta('1 days 2 hours 3 ns')
        result = td // np.timedelta64(1, 'D')
        assert result == 1
        result = td // np.timedelta64(1, 's')
        assert result == 93600
        result = td // np.timedelta64(1, 'ns')
        assert result == td._value

    def test_td_div_numeric_scalar(self):
        td = Timedelta(10, unit='d')
        result = td / 2
        assert isinstance(result, Timedelta)
        assert result == Timedelta(days=5)
        result = td / 5
        assert isinstance(result, Timedelta)
        assert result == Timedelta(days=2)

    @pytest.mark.parametrize('nan', [np.nan, np.float64('NaN'), float('nan')])
    def test_td_div_nan(self, nan):
        td = Timedelta(10, unit='d')
        result = td / nan
        assert result is NaT
        result = td // nan
        assert result is NaT

    def test_td_div_td64_ndarray(self):
        td = Timedelta('1 day')
        other = np.array([Timedelta('2 Days').to_timedelta64()])
        expected = np.array([0.5])
        result = td / other
        tm.assert_numpy_array_equal(result, expected)
        result = other / td
        tm.assert_numpy_array_equal(result, expected * 4)

    def test_td_div_ndarray_0d(self):
        td = Timedelta('1 day')
        other = np.array(1)
        res = td / other
        assert isinstance(res, Timedelta)
        assert res == td

    def test_td_rdiv_timedeltalike_scalar(self):
        td = Timedelta(10, unit='d')
        result = offsets.Hour(1) / td
        assert result == 1 / 240.0
        assert np.timedelta64(60, 'h') / td == 0.25

    def test_td_rdiv_na_scalar(self):
        td = Timedelta(10, unit='d')
        result = NaT / td
        assert np.isnan(result)
        result = None / td
        assert np.isnan(result)
        result = np.timedelta64('NaT') / td
        assert np.isnan(result)
        msg = "unsupported operand type\\(s\\) for /: 'numpy.datetime64' and 'Timedelta'"
        with pytest.raises(TypeError, match=msg):
            np.datetime64('NaT') / td
        msg = "unsupported operand type\\(s\\) for /: 'float' and 'Timedelta'"
        with pytest.raises(TypeError, match=msg):
            np.nan / td

    def test_td_rdiv_ndarray(self):
        td = Timedelta(10, unit='d')
        arr = np.array([td], dtype=object)
        result = arr / td
        expected = np.array([1], dtype=np.float64)
        tm.assert_numpy_array_equal(result, expected)
        arr = np.array([None])
        result = arr / td
        expected = np.array([np.nan])
        tm.assert_numpy_array_equal(result, expected)
        arr = np.array([np.nan], dtype=object)
        msg = "unsupported operand type\\(s\\) for /: 'float' and 'Timedelta'"
        with pytest.raises(TypeError, match=msg):
            arr / td
        arr = np.array([np.nan], dtype=np.float64)
        msg = 'cannot use operands with types dtype'
        with pytest.raises(TypeError, match=msg):
            arr / td

    def test_td_rdiv_ndarray_0d(self):
        td = Timedelta(10, unit='d')
        arr = np.array(td.asm8)
        assert arr / td == 1

    def test_td_floordiv_timedeltalike_scalar(self):
        td = Timedelta(hours=3, minutes=4)
        scalar = Timedelta(hours=3, minutes=3)
        assert td // scalar == 1
        assert -td // scalar.to_pytimedelta() == -2
        assert 2 * td // scalar.to_timedelta64() == 2

    def test_td_floordiv_null_scalar(self):
        td = Timedelta(hours=3, minutes=4)
        assert td // np.nan is NaT
        assert np.isnan(td // NaT)
        assert np.isnan(td // np.timedelta64('NaT'))

    def test_td_floordiv_offsets(self):
        td = Timedelta(hours=3, minutes=4)
        assert td // offsets.Hour(1) == 3
        assert td // offsets.Minute(2) == 92

    def test_td_floordiv_invalid_scalar(self):
        td = Timedelta(hours=3, minutes=4)
        msg = '|'.join(['Invalid dtype datetime64\\[D\\] for __floordiv__', "'dtype' is an invalid keyword argument for this function", "ufunc '?floor_divide'? cannot use operands with types"])
        with pytest.raises(TypeError, match=msg):
            td // np.datetime64('2016-01-01', dtype='datetime64[us]')

    def test_td_floordiv_numeric_scalar(self):
        td = Timedelta(hours=3, minutes=4)
        expected = Timedelta(hours=1, minutes=32)
        assert td // 2 == expected
        assert td // 2.0 == expected
        assert td // np.float64(2.0) == expected
        assert td // np.int32(2.0) == expected
        assert td // np.uint8(2.0) == expected

    def test_td_floordiv_timedeltalike_array(self):
        td = Timedelta(hours=3, minutes=4)
        scalar = Timedelta(hours=3, minutes=3)
        assert td // np.array(scalar.to_timedelta64()) == 1
        res = 3 * td // np.array([scalar.to_timedelta64()])
        expected = np.array([3], dtype=np.int64)
        tm.assert_numpy_array_equal(res, expected)
        res = 10 * td // np.array([scalar.to_timedelta64(), np.timedelta64('NaT')])
        expected = np.array([10, np.nan])
        tm.assert_numpy_array_equal(res, expected)

    def test_td_floordiv_numeric_series(self):
        td = Timedelta(hours=3, minutes=4)
        ser = pd.Series([1], dtype=np.int64)
        res = td // ser
        assert res.dtype.kind == 'm'

    def test_td_rfloordiv_timedeltalike_scalar(self):
        td = Timedelta(hours=3, minutes=3)
        scalar = Timedelta(hours=3, minutes=4)
        assert td.__rfloordiv__(scalar) == 1
        assert (-td).__rfloordiv__(scalar.to_pytimedelta()) == -2
        assert (2 * td).__rfloordiv__(scalar.to_timedelta64()) == 0

    def test_td_rfloordiv_null_scalar(self):
        td = Timedelta(hours=3, minutes=3)
        assert np.isnan(td.__rfloordiv__(NaT))
        assert np.isnan(td.__rfloordiv__(np.timedelta64('NaT')))

    def test_td_rfloordiv_offsets(self):
        assert offsets.Hour(1) // Timedelta(minutes=25) == 2

    def test_td_rfloordiv_invalid_scalar(self):
        td = Timedelta(hours=3, minutes=3)
        dt64 = np.datetime64('2016-01-01', 'us')
        assert td.__rfloordiv__(dt64) is NotImplemented
        msg = "unsupported operand type\\(s\\) for //: 'numpy.datetime64' and 'Timedelta'"
        with pytest.raises(TypeError, match=msg):
            dt64 // td

    def test_td_rfloordiv_numeric_scalar(self):
        td = Timedelta(hours=3, minutes=3)
        assert td.__rfloordiv__(np.nan) is NotImplemented
        assert td.__rfloordiv__(3.5) is NotImplemented
        assert td.__rfloordiv__(2) is NotImplemented
        assert td.__rfloordiv__(np.float64(2.0)) is NotImplemented
        assert td.__rfloordiv__(np.uint8(9)) is NotImplemented
        assert td.__rfloordiv__(np.int32(2.0)) is NotImplemented
        msg = "unsupported operand type\\(s\\) for //: '.*' and 'Timedelta"
        with pytest.raises(TypeError, match=msg):
            np.float64(2.0) // td
        with pytest.raises(TypeError, match=msg):
            np.uint8(9) // td
        with pytest.raises(TypeError, match=msg):
            np.int32(2.0) // td

    def test_td_rfloordiv_timedeltalike_array(self):
        td = Timedelta(hours=3, minutes=3)
        scalar = Timedelta(hours=3, minutes=4)
        assert td.__rfloordiv__(np.array(scalar.to_timedelta64())) == 1
        res = td.__rfloordiv__(np.array([(3 * scalar).to_timedelta64()]))
        expected = np.array([3], dtype=np.int64)
        tm.assert_numpy_array_equal(res, expected)
        arr = np.array([(10 * scalar).to_timedelta64(), np.timedelta64('NaT')])
        res = td.__rfloordiv__(arr)
        expected = np.array([10, np.nan])
        tm.assert_numpy_array_equal(res, expected)

    def test_td_rfloordiv_intarray(self):
        ints = np.array([1349654400, 1349740800, 1349827200, 1349913600]) * 10 ** 9
        msg = 'Invalid dtype'
        with pytest.raises(TypeError, match=msg):
            ints // Timedelta(1, unit='s')

    def test_td_rfloordiv_numeric_series(self):
        td = Timedelta(hours=3, minutes=3)
        ser = pd.Series([1], dtype=np.int64)
        res = td.__rfloordiv__(ser)
        assert res is NotImplemented
        msg = 'Invalid dtype'
        with pytest.raises(TypeError, match=msg):
            ser // td

    def test_mod_timedeltalike(self):
        td = Timedelta(hours=37)
        result = td % Timedelta(hours=6)
        assert isinstance(result, Timedelta)
        assert result == Timedelta(hours=1)
        result = td % timedelta(minutes=60)
        assert isinstance(result, Timedelta)
        assert result == Timedelta(0)
        result = td % NaT
        assert result is NaT

    def test_mod_timedelta64_nat(self):
        td = Timedelta(hours=37)
        result = td % np.timedelta64('NaT', 'ns')
        assert result is NaT

    def test_mod_timedelta64(self):
        td = Timedelta(hours=37)
        result = td % np.timedelta64(2, 'h')
        assert isinstance(result, Timedelta)
        assert result == Timedelta(hours=1)

    def test_mod_offset(self):
        td = Timedelta(hours=37)
        result = td % offsets.Hour(5)
        assert isinstance(result, Timedelta)
        assert result == Timedelta(hours=2)

    def test_mod_numeric(self):
        td = Timedelta(hours=37)
        result = td % 2
        assert isinstance(result, Timedelta)
        assert result == Timedelta(0)
        result = td % 1000000000000.0
        assert isinstance(result, Timedelta)
        assert result == Timedelta(minutes=3, seconds=20)
        result = td % int(1000000000000.0)
        assert isinstance(result, Timedelta)
        assert result == Timedelta(minutes=3, seconds=20)

    def test_mod_invalid(self):
        td = Timedelta(hours=37)
        msg = 'unsupported operand type'
        with pytest.raises(TypeError, match=msg):
            td % Timestamp('2018-01-22')
        with pytest.raises(TypeError, match=msg):
            td % []

    def test_rmod_pytimedelta(self):
        td = Timedelta(minutes=3)
        result = timedelta(minutes=4) % td
        assert isinstance(result, Timedelta)
        assert result == Timedelta(minutes=1)

    def test_rmod_timedelta64(self):
        td = Timedelta(minutes=3)
        result = np.timedelta64(5, 'm') % td
        assert isinstance(result, Timedelta)
        assert result == Timedelta(minutes=2)

    def test_rmod_invalid(self):
        td = Timedelta(minutes=3)
        msg = 'unsupported operand'
        with pytest.raises(TypeError, match=msg):
            Timestamp('2018-01-22') % td
        with pytest.raises(TypeError, match=msg):
            15 % td
        with pytest.raises(TypeError, match=msg):
            16.0 % td
        msg = 'Invalid dtype int'
        with pytest.raises(TypeError, match=msg):
            np.array([22, 24]) % td

    def test_divmod_numeric(self):
        td = Timedelta(days=2, hours=6)
        result = divmod(td, 53 * 3600 * 1000000000.0)
        assert result[0] == Timedelta(1, unit='ns')
        assert isinstance(result[1], Timedelta)
        assert result[1] == Timedelta(hours=1)
        assert result
        result = divmod(td, np.nan)
        assert result[0] is NaT
        assert result[1] is NaT

    def test_divmod(self):
        td = Timedelta(days=2, hours=6)
        result = divmod(td, timedelta(days=1))
        assert result[0] == 2
        assert isinstance(result[1], Timedelta)
        assert result[1] == Timedelta(hours=6)
        result = divmod(td, 54)
        assert result[0] == Timedelta(hours=1)
        assert isinstance(result[1], Timedelta)
        assert result[1] == Timedelta(0)
        result = divmod(td, NaT)
        assert np.isnan(result[0])
        assert result[1] is NaT

    def test_divmod_offset(self):
        td = Timedelta(days=2, hours=6)
        result = divmod(td, offsets.Hour(-4))
        assert result[0] == -14
        assert isinstance(result[1], Timedelta)
        assert result[1] == Timedelta(hours=-2)

    def test_divmod_invalid(self):
        td = Timedelta(days=2, hours=6)
        msg = "unsupported operand type\\(s\\) for //: 'Timedelta' and 'Timestamp'"
        with pytest.raises(TypeError, match=msg):
            divmod(td, Timestamp('2018-01-22'))

    def test_rdivmod_pytimedelta(self):
        result = divmod(timedelta(days=2, hours=6), Timedelta(days=1))
        assert result[0] == 2
        assert isinstance(result[1], Timedelta)
        assert result[1] == Timedelta(hours=6)

    def test_rdivmod_offset(self):
        result = divmod(offsets.Hour(54), Timedelta(hours=-4))
        assert result[0] == -14
        assert isinstance(result[1], Timedelta)
        assert result[1] == Timedelta(hours=-2)

    def test_rdivmod_invalid(self):
        td = Timedelta(minutes=3)
        msg = 'unsupported operand type'
        with pytest.raises(TypeError, match=msg):
            divmod(Timestamp('2018-01-22'), td)
        with pytest.raises(TypeError, match=msg):
            divmod(15, td)
        with pytest.raises(TypeError, match=msg):
            divmod(16.0, td)
        msg = 'Invalid dtype int'
        with pytest.raises(TypeError, match=msg):
            divmod(np.array([22, 24]), td)

    @pytest.mark.parametrize('op', [operator.mul, ops.rmul, operator.truediv, ops.rdiv, ops.rsub])
    @pytest.mark.parametrize('arr', [np.array([Timestamp('20130101 9:01'), Timestamp('20121230 9:02')]), np.array([Timestamp('2021-11-09 09:54:00'), Timedelta('1D')])])
    def test_td_op_timedelta_timedeltalike_array(self, op, arr):
        msg = 'unsupported operand type|cannot use operands with types'
        with pytest.raises(TypeError, match=msg):
            op(arr, Timedelta('1D'))