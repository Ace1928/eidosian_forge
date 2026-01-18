from datetime import datetime
import pytest
from pandas._libs.tslibs.ccalendar import (
import pandas._libs.tslibs.offsets as liboffsets
from pandas._libs.tslibs.offsets import roll_qtrday
from pandas import Timestamp
@pytest.mark.parametrize('other,expected', [(datetime(2017, 2, 10), {2: 1, -7: -7, 0: 0}), (Timestamp('2014-03-15', tz='US/Eastern'), {2: 2, -7: -6, 0: 1})])
@pytest.mark.parametrize('n', [2, -7, 0])
def test_roll_qtrday_year(other, expected, n):
    month = 3
    day_opt = 'start'
    assert roll_qtrday(other, n, month, day_opt, modby=12) == expected[n]