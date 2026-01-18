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
@pytest.mark.parametrize('day', ['1970/01/01 ', '2020-12-31 ', '1981/09/13 '])
@pytest.mark.parametrize('hour', ['00:00:00', '00:00:01', '23:59:59', '12:00:59'])
@pytest.mark.parametrize('sec_float, expected', [('.000000001', 1), ('.000000999', 999), ('.123456789', 789), ('.999999999', 999), ('.999999000', 0), ('.999999001123', 1), ('.999999001123456', 1), ('.999999001123456789', 1)])
def test_period_constructor_nanosecond(self, day, hour, sec_float, expected):
    assert Period(day + hour + sec_float).start_time.nanosecond == expected