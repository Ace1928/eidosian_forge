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
def test_properties_quarterly(self):
    qedec_date = Period(freq='Q-DEC', year=2007, quarter=1)
    qejan_date = Period(freq='Q-JAN', year=2007, quarter=1)
    qejun_date = Period(freq='Q-JUN', year=2007, quarter=1)
    for x in range(3):
        for qd in (qedec_date, qejan_date, qejun_date):
            assert (qd + x).qyear == 2007
            assert (qd + x).quarter == x + 1