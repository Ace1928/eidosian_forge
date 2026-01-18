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
def test_period_from_ordinal(self):
    p = Period('2011-01', freq='M')
    res = Period._from_ordinal(p.ordinal, freq=p.freq)
    assert p == res
    assert isinstance(res, Period)