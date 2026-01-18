import dateutil.tz
from dateutil.tz import tzlocal
import pytest
import pytz
from pandas._libs.tslibs.ccalendar import MONTHS
from pandas._libs.tslibs.offsets import MonthEnd
from pandas._libs.tslibs.period import INVALID_FREQ_ERR_MSG
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('freq, freq_depr', [('2ME', '2M'), ('2QE', '2Q'), ('2QE-SEP', '2Q-SEP'), ('1YE', '1Y'), ('2YE-MAR', '2Y-MAR'), ('1YE', '1A'), ('2YE-MAR', '2A-MAR')])
def test_to_period_frequency_M_Q_Y_A_deprecated(self, freq, freq_depr):
    msg = f"'{freq_depr[1:]}' is deprecated and will be removed "
    f"in a future version, please use '{freq[1:]}' instead."
    rng = date_range('01-Jan-2012', periods=8, freq=freq)
    prng = rng.to_period()
    with tm.assert_produces_warning(FutureWarning, match=msg):
        assert prng.freq == freq_depr