from datetime import datetime
from dateutil.tz import gettz
from hypothesis import (
import numpy as np
import pytest
import pytz
from pytz import utc
from pandas._libs import lib
from pandas._libs.tslibs import (
from pandas._libs.tslibs.dtypes import NpyDatetimeUnit
from pandas._libs.tslibs.period import INVALID_FREQ_ERR_MSG
import pandas.util._test_decorators as td
import pandas._testing as tm
@pytest.mark.parametrize('unit', ['ns', 'us', 'ms', 's'])
def test_replace_dst_border(self, unit):
    t = Timestamp('2013-11-3', tz='America/Chicago').as_unit(unit)
    result = t.replace(hour=3)
    expected = Timestamp('2013-11-3 03:00:00', tz='America/Chicago')
    assert result == expected
    assert result._creso == getattr(NpyDatetimeUnit, f'NPY_FR_{unit}').value