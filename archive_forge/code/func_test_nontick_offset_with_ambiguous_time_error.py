from datetime import timedelta
import pytest
import pytz
from pandas._libs.tslibs import Timestamp
from pandas._libs.tslibs.offsets import (
from pandas.errors import PerformanceWarning
from pandas import DatetimeIndex
import pandas._testing as tm
from pandas.util.version import Version
@pytest.mark.parametrize('original_dt, target_dt, offset, tz', [pytest.param(Timestamp('1900-01-01'), Timestamp('1905-07-01'), MonthBegin(66), 'Africa/Lagos', marks=pytest.mark.xfail(pytz_version < Version('2020.5') or pytz_version == Version('2022.2'), reason='GH#41906: pytz utc transition dates changed')), (Timestamp('2021-10-01 01:15'), Timestamp('2021-10-31 01:15'), MonthEnd(1), 'Europe/London'), (Timestamp('2010-12-05 02:59'), Timestamp('2010-10-31 02:59'), SemiMonthEnd(-3), 'Europe/Paris'), (Timestamp('2021-10-31 01:20'), Timestamp('2021-11-07 01:20'), CustomBusinessDay(2, weekmask='Sun Mon'), 'US/Eastern'), (Timestamp('2020-04-03 01:30'), Timestamp('2020-11-01 01:30'), YearBegin(1, month=11), 'America/Chicago')])
def test_nontick_offset_with_ambiguous_time_error(original_dt, target_dt, offset, tz):
    localized_dt = original_dt.tz_localize(tz)
    msg = f"Cannot infer dst time from {target_dt}, try using the 'ambiguous' argument"
    with pytest.raises(pytz.AmbiguousTimeError, match=msg):
        localized_dt + offset