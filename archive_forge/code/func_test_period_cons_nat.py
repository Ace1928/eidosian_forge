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
def test_period_cons_nat(self):
    p = Period('nat', freq='W-SUN')
    assert p is NaT
    p = Period(iNaT, freq='D')
    assert p is NaT
    p = Period(iNaT, freq='3D')
    assert p is NaT
    p = Period(iNaT, freq='1D1h')
    assert p is NaT
    p = Period('NaT')
    assert p is NaT
    p = Period(iNaT)
    assert p is NaT