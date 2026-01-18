import dateutil.tz
from dateutil.tz import tzlocal
import pytest
import pytz
from pandas._libs.tslibs.ccalendar import MONTHS
from pandas._libs.tslibs.offsets import MonthEnd
from pandas._libs.tslibs.period import INVALID_FREQ_ERR_MSG
from pandas import (
import pandas._testing as tm
def test_to_period_nofreq(self):
    idx = DatetimeIndex(['2000-01-01', '2000-01-02', '2000-01-04'])
    msg = 'You must pass a freq argument as current index has none.'
    with pytest.raises(ValueError, match=msg):
        idx.to_period()
    idx = DatetimeIndex(['2000-01-01', '2000-01-02', '2000-01-03'], freq='infer')
    assert idx.freqstr == 'D'
    expected = PeriodIndex(['2000-01-01', '2000-01-02', '2000-01-03'], freq='D')
    tm.assert_index_equal(idx.to_period(), expected)
    idx = DatetimeIndex(['2000-01-01', '2000-01-02', '2000-01-03'])
    assert idx.freqstr is None
    tm.assert_index_equal(idx.to_period(), expected)