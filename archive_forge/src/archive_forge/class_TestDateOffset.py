from __future__ import annotations
from datetime import (
import numpy as np
import pytest
from pandas._libs.tslibs import (
import pandas._libs.tslibs.offsets as liboffsets
from pandas._libs.tslibs.offsets import (
from pandas._libs.tslibs.period import INVALID_FREQ_ERR_MSG
from pandas.errors import PerformanceWarning
from pandas import (
import pandas._testing as tm
from pandas.tests.tseries.offsets.common import WeekDay
from pandas.tseries import offsets
from pandas.tseries.offsets import (
class TestDateOffset:

    def setup_method(self):
        _offset_map.clear()

    def test_repr(self):
        repr(DateOffset())
        repr(DateOffset(2))
        repr(2 * DateOffset())
        repr(2 * DateOffset(months=2))

    def test_mul(self):
        assert DateOffset(2) == 2 * DateOffset(1)
        assert DateOffset(2) == DateOffset(1) * 2

    @pytest.mark.parametrize('kwd', sorted(liboffsets._relativedelta_kwds))
    def test_constructor(self, kwd, request):
        if kwd == 'millisecond':
            request.applymarker(pytest.mark.xfail(raises=NotImplementedError, reason='Constructing DateOffset object with `millisecond` is not yet supported.'))
        offset = DateOffset(**{kwd: 2})
        assert offset.kwds == {kwd: 2}
        assert getattr(offset, kwd) == 2

    def test_default_constructor(self, dt):
        assert dt + DateOffset(2) == datetime(2008, 1, 4)

    def test_is_anchored(self):
        msg = 'DateOffset.is_anchored is deprecated '
        with tm.assert_produces_warning(FutureWarning, match=msg):
            assert not DateOffset(2).is_anchored()
            assert DateOffset(1).is_anchored()

    def test_copy(self):
        assert DateOffset(months=2).copy() == DateOffset(months=2)
        assert DateOffset(milliseconds=1).copy() == DateOffset(milliseconds=1)

    @pytest.mark.parametrize('arithmatic_offset_type, expected', zip(_ARITHMETIC_DATE_OFFSET, ['2009-01-02', '2008-02-02', '2008-01-09', '2008-01-03', '2008-01-02 01:00:00', '2008-01-02 00:01:00', '2008-01-02 00:00:01', '2008-01-02 00:00:00.001000000', '2008-01-02 00:00:00.000001000']))
    def test_add(self, arithmatic_offset_type, expected, dt):
        assert DateOffset(**{arithmatic_offset_type: 1}) + dt == Timestamp(expected)
        assert dt + DateOffset(**{arithmatic_offset_type: 1}) == Timestamp(expected)

    @pytest.mark.parametrize('arithmatic_offset_type, expected', zip(_ARITHMETIC_DATE_OFFSET, ['2007-01-02', '2007-12-02', '2007-12-26', '2008-01-01', '2008-01-01 23:00:00', '2008-01-01 23:59:00', '2008-01-01 23:59:59', '2008-01-01 23:59:59.999000000', '2008-01-01 23:59:59.999999000']))
    def test_sub(self, arithmatic_offset_type, expected, dt):
        assert dt - DateOffset(**{arithmatic_offset_type: 1}) == Timestamp(expected)
        with pytest.raises(TypeError, match='Cannot subtract datetime from offset'):
            DateOffset(**{arithmatic_offset_type: 1}) - dt

    @pytest.mark.parametrize('arithmatic_offset_type, n, expected', zip(_ARITHMETIC_DATE_OFFSET, range(1, 10), ['2009-01-02', '2008-03-02', '2008-01-23', '2008-01-06', '2008-01-02 05:00:00', '2008-01-02 00:06:00', '2008-01-02 00:00:07', '2008-01-02 00:00:00.008000000', '2008-01-02 00:00:00.000009000']))
    def test_mul_add(self, arithmatic_offset_type, n, expected, dt):
        assert DateOffset(**{arithmatic_offset_type: 1}) * n + dt == Timestamp(expected)
        assert n * DateOffset(**{arithmatic_offset_type: 1}) + dt == Timestamp(expected)
        assert dt + DateOffset(**{arithmatic_offset_type: 1}) * n == Timestamp(expected)
        assert dt + n * DateOffset(**{arithmatic_offset_type: 1}) == Timestamp(expected)

    @pytest.mark.parametrize('arithmatic_offset_type, n, expected', zip(_ARITHMETIC_DATE_OFFSET, range(1, 10), ['2007-01-02', '2007-11-02', '2007-12-12', '2007-12-29', '2008-01-01 19:00:00', '2008-01-01 23:54:00', '2008-01-01 23:59:53', '2008-01-01 23:59:59.992000000', '2008-01-01 23:59:59.999991000']))
    def test_mul_sub(self, arithmatic_offset_type, n, expected, dt):
        assert dt - DateOffset(**{arithmatic_offset_type: 1}) * n == Timestamp(expected)
        assert dt - n * DateOffset(**{arithmatic_offset_type: 1}) == Timestamp(expected)

    def test_leap_year(self):
        d = datetime(2008, 1, 31)
        assert d + DateOffset(months=1) == datetime(2008, 2, 29)

    def test_eq(self):
        offset1 = DateOffset(days=1)
        offset2 = DateOffset(days=365)
        assert offset1 != offset2
        assert DateOffset(milliseconds=3) != DateOffset(milliseconds=7)

    @pytest.mark.parametrize('offset_kwargs, expected_arg', [({'microseconds': 1, 'milliseconds': 1}, '2022-01-01 00:00:00.001001'), ({'seconds': 1, 'milliseconds': 1}, '2022-01-01 00:00:01.001'), ({'minutes': 1, 'milliseconds': 1}, '2022-01-01 00:01:00.001'), ({'hours': 1, 'milliseconds': 1}, '2022-01-01 01:00:00.001'), ({'days': 1, 'milliseconds': 1}, '2022-01-02 00:00:00.001'), ({'weeks': 1, 'milliseconds': 1}, '2022-01-08 00:00:00.001'), ({'months': 1, 'milliseconds': 1}, '2022-02-01 00:00:00.001'), ({'years': 1, 'milliseconds': 1}, '2023-01-01 00:00:00.001')])
    def test_milliseconds_combination(self, offset_kwargs, expected_arg):
        offset = DateOffset(**offset_kwargs)
        ts = Timestamp('2022-01-01')
        result = ts + offset
        expected = Timestamp(expected_arg)
        assert result == expected

    def test_offset_invalid_arguments(self):
        msg = '^Invalid argument/s or bad combination of arguments'
        with pytest.raises(ValueError, match=msg):
            DateOffset(picoseconds=1)