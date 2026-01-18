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
def test_period_immutable():
    msg = 'not writable'
    per = Period('2014Q1')
    with pytest.raises(AttributeError, match=msg):
        per.ordinal = 14
    freq = per.freq
    with pytest.raises(AttributeError, match=msg):
        per.freq = 2 * freq