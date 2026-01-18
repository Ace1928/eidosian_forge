import dateutil.tz
from dateutil.tz import tzlocal
import pytest
import pytz
from pandas._libs.tslibs.ccalendar import MONTHS
from pandas._libs.tslibs.offsets import MonthEnd
from pandas._libs.tslibs.period import INVALID_FREQ_ERR_MSG
from pandas import (
import pandas._testing as tm
def test_to_period_monthish(self):
    offsets = ['MS', 'BME']
    for off in offsets:
        rng = date_range('01-Jan-2012', periods=8, freq=off)
        prng = rng.to_period()
        assert prng.freqstr == 'M'
    rng = date_range('01-Jan-2012', periods=8, freq='ME')
    prng = rng.to_period()
    assert prng.freqstr == 'M'
    with pytest.raises(ValueError, match=INVALID_FREQ_ERR_MSG):
        date_range('01-Jan-2012', periods=8, freq='EOM')