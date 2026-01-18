from datetime import datetime
import pytest
from pandas._libs.tslibs.ccalendar import (
import pandas._libs.tslibs.offsets as liboffsets
from pandas._libs.tslibs.offsets import roll_qtrday
from pandas import Timestamp
@pytest.mark.parametrize('other,month,exp_dict', [(datetime(1999, 5, 31), 2, {-1: {'start': 0, 'business_start': 0}}), (Timestamp(2072, 10, 1, 6, 17, 18), 4, {2: {'end': 1, 'business_end': 1, 'business_start': 1}}), (Timestamp(2072, 10, 3, 6, 17, 18), 4, {2: {'end': 1, 'business_end': 1}, -1: {'start': 0}})])
@pytest.mark.parametrize('n', [2, -1])
def test_roll_qtr_day_mod_equal(other, month, exp_dict, n, day_opt):
    expected = exp_dict.get(n, {}).get(day_opt, n)
    assert roll_qtrday(other, n, month, day_opt, modby=3) == expected