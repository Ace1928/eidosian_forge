from datetime import (
import re
import numpy as np
import pytest
from pandas._libs.tslibs import iNaT
from pandas._libs.tslibs.ccalendar import (
from pandas._libs.tslibs.np_datetime import OutOfBoundsDatetime
from pandas._libs.tslibs.parsing import DateParseError
from pandas._libs.tslibs.period import INVALID_FREQ_ERR_MSG
from pandas import (
import pandas._testing as tm
def test_period_constructor_offsets(self):
    assert Period('1/1/2005', freq=offsets.MonthEnd()) == Period('1/1/2005', freq='M')
    assert Period('2005', freq=offsets.YearEnd()) == Period('2005', freq='Y')
    assert Period('2005', freq=offsets.MonthEnd()) == Period('2005', freq='M')
    with tm.assert_produces_warning(FutureWarning, match=bday_msg):
        assert Period('3/10/12', freq=offsets.BusinessDay()) == Period('3/10/12', freq='B')
    assert Period('3/10/12', freq=offsets.Day()) == Period('3/10/12', freq='D')
    assert Period(year=2005, quarter=1, freq=offsets.QuarterEnd(startingMonth=12)) == Period(year=2005, quarter=1, freq='Q')
    assert Period(year=2005, quarter=2, freq=offsets.QuarterEnd(startingMonth=12)) == Period(year=2005, quarter=2, freq='Q')
    assert Period(year=2005, month=3, day=1, freq=offsets.Day()) == Period(year=2005, month=3, day=1, freq='D')
    with tm.assert_produces_warning(FutureWarning, match=bday_msg):
        assert Period(year=2012, month=3, day=10, freq=offsets.BDay()) == Period(year=2012, month=3, day=10, freq='B')
    expected = Period('2005-03-01', freq='3D')
    assert Period(year=2005, month=3, day=1, freq=offsets.Day(3)) == expected
    assert Period(year=2005, month=3, day=1, freq='3D') == expected
    with tm.assert_produces_warning(FutureWarning, match=bday_msg):
        assert Period(year=2012, month=3, day=10, freq=offsets.BDay(3)) == Period(year=2012, month=3, day=10, freq='3B')
    assert Period(200701, freq=offsets.MonthEnd()) == Period(200701, freq='M')
    i1 = Period(ordinal=200701, freq=offsets.MonthEnd())
    i2 = Period(ordinal=200701, freq='M')
    assert i1 == i2
    assert i1.year == 18695
    assert i2.year == 18695
    i1 = Period(datetime(2007, 1, 1), freq='M')
    i2 = Period('200701', freq='M')
    assert i1 == i2
    i1 = Period(date(2007, 1, 1), freq='M')
    i2 = Period(datetime(2007, 1, 1), freq='M')
    i3 = Period(np.datetime64('2007-01-01'), freq='M')
    i4 = Period('2007-01-01 00:00:00', freq='M')
    i5 = Period('2007-01-01 00:00:00.000', freq='M')
    assert i1 == i2
    assert i1 == i3
    assert i1 == i4
    assert i1 == i5
    i1 = Period('2007-01-01 09:00:00.001')
    expected = Period(datetime(2007, 1, 1, 9, 0, 0, 1000), freq='ms')
    assert i1 == expected
    expected = Period('2007-01-01 09:00:00.001', freq='ms')
    assert i1 == expected
    i1 = Period('2007-01-01 09:00:00.00101')
    expected = Period(datetime(2007, 1, 1, 9, 0, 0, 1010), freq='us')
    assert i1 == expected
    expected = Period('2007-01-01 09:00:00.00101', freq='us')
    assert i1 == expected