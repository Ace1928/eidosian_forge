from datetime import timedelta
import numpy as np
import pytest
from pandas._libs.tslibs.period import IncompatibleFrequency
from pandas import (
def test_period_add_offset(self):
    for freq in ['Y', '2Y', '3Y']:
        per = Period('2011', freq=freq)
        exp = Period('2013', freq=freq)
        assert per + offsets.YearEnd(2) == exp
        assert offsets.YearEnd(2) + per == exp
        for off in [offsets.YearBegin(2), offsets.MonthBegin(1), offsets.Minute(), np.timedelta64(365, 'D'), timedelta(365)]:
            msg = 'Input has different freq|Input cannot be converted to Period'
            with pytest.raises(IncompatibleFrequency, match=msg):
                per + off
            with pytest.raises(IncompatibleFrequency, match=msg):
                off + per
    for freq in ['M', '2M', '3M']:
        per = Period('2011-03', freq=freq)
        exp = Period('2011-05', freq=freq)
        assert per + offsets.MonthEnd(2) == exp
        assert offsets.MonthEnd(2) + per == exp
        exp = Period('2012-03', freq=freq)
        assert per + offsets.MonthEnd(12) == exp
        assert offsets.MonthEnd(12) + per == exp
        msg = '|'.join(['Input has different freq', 'Input cannot be converted to Period'])
        for off in [offsets.YearBegin(2), offsets.MonthBegin(1), offsets.Minute(), np.timedelta64(365, 'D'), timedelta(365)]:
            with pytest.raises(IncompatibleFrequency, match=msg):
                per + off
            with pytest.raises(IncompatibleFrequency, match=msg):
                off + per
    for freq in ['D', '2D', '3D']:
        per = Period('2011-04-01', freq=freq)
        exp = Period('2011-04-06', freq=freq)
        assert per + offsets.Day(5) == exp
        assert offsets.Day(5) + per == exp
        exp = Period('2011-04-02', freq=freq)
        assert per + offsets.Hour(24) == exp
        assert offsets.Hour(24) + per == exp
        exp = Period('2011-04-03', freq=freq)
        assert per + np.timedelta64(2, 'D') == exp
        assert np.timedelta64(2, 'D') + per == exp
        exp = Period('2011-04-02', freq=freq)
        assert per + np.timedelta64(3600 * 24, 's') == exp
        assert np.timedelta64(3600 * 24, 's') + per == exp
        exp = Period('2011-03-30', freq=freq)
        assert per + timedelta(-2) == exp
        assert timedelta(-2) + per == exp
        exp = Period('2011-04-03', freq=freq)
        assert per + timedelta(hours=48) == exp
        assert timedelta(hours=48) + per == exp
        msg = '|'.join(['Input has different freq', 'Input cannot be converted to Period'])
        for off in [offsets.YearBegin(2), offsets.MonthBegin(1), offsets.Minute(), np.timedelta64(4, 'h'), timedelta(hours=23)]:
            with pytest.raises(IncompatibleFrequency, match=msg):
                per + off
            with pytest.raises(IncompatibleFrequency, match=msg):
                off + per
    for freq in ['h', '2h', '3h']:
        per = Period('2011-04-01 09:00', freq=freq)
        exp = Period('2011-04-03 09:00', freq=freq)
        assert per + offsets.Day(2) == exp
        assert offsets.Day(2) + per == exp
        exp = Period('2011-04-01 12:00', freq=freq)
        assert per + offsets.Hour(3) == exp
        assert offsets.Hour(3) + per == exp
        msg = 'cannot use operands with types'
        exp = Period('2011-04-01 12:00', freq=freq)
        assert per + np.timedelta64(3, 'h') == exp
        assert np.timedelta64(3, 'h') + per == exp
        exp = Period('2011-04-01 10:00', freq=freq)
        assert per + np.timedelta64(3600, 's') == exp
        assert np.timedelta64(3600, 's') + per == exp
        exp = Period('2011-04-01 11:00', freq=freq)
        assert per + timedelta(minutes=120) == exp
        assert timedelta(minutes=120) + per == exp
        exp = Period('2011-04-05 12:00', freq=freq)
        assert per + timedelta(days=4, minutes=180) == exp
        assert timedelta(days=4, minutes=180) + per == exp
        msg = '|'.join(['Input has different freq', 'Input cannot be converted to Period'])
        for off in [offsets.YearBegin(2), offsets.MonthBegin(1), offsets.Minute(), np.timedelta64(3200, 's'), timedelta(hours=23, minutes=30)]:
            with pytest.raises(IncompatibleFrequency, match=msg):
                per + off
            with pytest.raises(IncompatibleFrequency, match=msg):
                off + per