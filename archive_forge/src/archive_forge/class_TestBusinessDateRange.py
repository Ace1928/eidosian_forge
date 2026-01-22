from datetime import (
import re
import numpy as np
import pytest
import pytz
from pytz import timezone
from pandas._libs.tslibs import timezones
from pandas._libs.tslibs.offsets import (
from pandas.errors import OutOfBoundsDatetime
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays.datetimes import _generate_range as generate_range
from pandas.tests.indexes.datetimes.test_timezones import (
from pandas.tseries.holiday import USFederalHolidayCalendar
class TestBusinessDateRange:

    def test_constructor(self):
        bdate_range(START, END, freq=BDay())
        bdate_range(START, periods=20, freq=BDay())
        bdate_range(end=START, periods=20, freq=BDay())
        msg = 'periods must be a number, got B'
        with pytest.raises(TypeError, match=msg):
            date_range('2011-1-1', '2012-1-1', 'B')
        with pytest.raises(TypeError, match=msg):
            bdate_range('2011-1-1', '2012-1-1', 'B')
        msg = 'freq must be specified for bdate_range; use date_range instead'
        with pytest.raises(TypeError, match=msg):
            bdate_range(START, END, periods=10, freq=None)

    def test_misc(self):
        end = datetime(2009, 5, 13)
        dr = bdate_range(end=end, periods=20)
        firstDate = end - 19 * BDay()
        assert len(dr) == 20
        assert dr[0] == firstDate
        assert dr[-1] == end

    def test_date_parse_failure(self):
        badly_formed_date = '2007/100/1'
        msg = 'Unknown datetime string format, unable to parse: 2007/100/1'
        with pytest.raises(ValueError, match=msg):
            Timestamp(badly_formed_date)
        with pytest.raises(ValueError, match=msg):
            bdate_range(start=badly_formed_date, periods=10)
        with pytest.raises(ValueError, match=msg):
            bdate_range(end=badly_formed_date, periods=10)
        with pytest.raises(ValueError, match=msg):
            bdate_range(badly_formed_date, badly_formed_date)

    def test_daterange_bug_456(self):
        rng1 = bdate_range('12/5/2011', '12/5/2011')
        rng2 = bdate_range('12/2/2011', '12/5/2011')
        assert rng2._data.freq == BDay()
        result = rng1.union(rng2)
        assert isinstance(result, DatetimeIndex)

    @pytest.mark.parametrize('inclusive', ['left', 'right', 'neither', 'both'])
    def test_bdays_and_open_boundaries(self, inclusive):
        start = '2018-07-21'
        end = '2018-07-29'
        result = date_range(start, end, freq='B', inclusive=inclusive)
        bday_start = '2018-07-23'
        bday_end = '2018-07-27'
        expected = date_range(bday_start, bday_end, freq='D')
        tm.assert_index_equal(result, expected)

    def test_bday_near_overflow(self):
        start = Timestamp.max.floor('D').to_pydatetime()
        rng = date_range(start, end=None, periods=1, freq='B')
        expected = DatetimeIndex([start], freq='B').as_unit('ns')
        tm.assert_index_equal(rng, expected)

    def test_bday_overflow_error(self):
        msg = 'Out of bounds nanosecond timestamp'
        start = Timestamp.max.floor('D').to_pydatetime()
        with pytest.raises(OutOfBoundsDatetime, match=msg):
            date_range(start, periods=2, freq='B')