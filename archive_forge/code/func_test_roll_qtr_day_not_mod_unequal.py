from datetime import datetime
import pytest
from pandas._libs.tslibs.ccalendar import (
import pandas._libs.tslibs.offsets as liboffsets
from pandas._libs.tslibs.offsets import roll_qtrday
from pandas import Timestamp
@pytest.mark.parametrize('month', [3, 5])
@pytest.mark.parametrize('n', [4, -3])
def test_roll_qtr_day_not_mod_unequal(day_opt, month, n):
    expected = {3: {-3: -2, 4: 4}, 5: {-3: -3, 4: 3}}
    other = Timestamp(2072, 10, 1, 6, 17, 18)
    assert roll_qtrday(other, n, month, day_opt, modby=3) == expected[month][n]