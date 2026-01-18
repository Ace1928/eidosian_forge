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
def test_parse_week_str_roundstrip(self):
    per = Period('2017-01-23/2017-01-29')
    assert per.freq.freqstr == 'W-SUN'
    per = Period('2017-01-24/2017-01-30')
    assert per.freq.freqstr == 'W-MON'
    msg = 'Could not parse as weekly-freq Period'
    with pytest.raises(ValueError, match=msg):
        Period('2016-01-23/2017-01-29')