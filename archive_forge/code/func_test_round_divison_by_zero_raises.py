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
def test_round_divison_by_zero_raises(self):
    ts = Timestamp('2016-01-01')
    msg = 'Division by zero in rounding'
    with pytest.raises(ValueError, match=msg):
        ts.round('0ns')