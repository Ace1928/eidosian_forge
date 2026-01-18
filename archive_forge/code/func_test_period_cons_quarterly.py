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
@pytest.mark.parametrize('month', MONTHS)
def test_period_cons_quarterly(self, month):
    freq = f'Q-{month}'
    exp = Period('1989Q3', freq=freq)
    assert '1989Q3' in str(exp)
    stamp = exp.to_timestamp('D', how='end')
    p = Period(stamp, freq=freq)
    assert p == exp
    stamp = exp.to_timestamp('3D', how='end')
    p = Period(stamp, freq=freq)
    assert p == exp