import dateutil.tz
from dateutil.tz import tzlocal
import pytest
import pytz
from pandas._libs.tslibs.ccalendar import MONTHS
from pandas._libs.tslibs.offsets import MonthEnd
from pandas._libs.tslibs.period import INVALID_FREQ_ERR_MSG
from pandas import (
import pandas._testing as tm
@pytest.mark.filterwarnings('ignore:PeriodDtype\\[B\\] is deprecated:FutureWarning')
def test_period_dt64_round_trip(self):
    dti = date_range('1/1/2000', '1/7/2002', freq='B')
    pi = dti.to_period()
    tm.assert_index_equal(pi.to_timestamp(), dti)
    dti = date_range('1/1/2000', '1/7/2002', freq='B')
    pi = dti.to_period(freq='h')
    tm.assert_index_equal(pi.to_timestamp(), dti)