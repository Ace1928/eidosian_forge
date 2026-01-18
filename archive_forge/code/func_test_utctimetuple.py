import calendar
from datetime import (
import locale
import time
import unicodedata
from dateutil.tz import (
from hypothesis import (
import numpy as np
import pytest
import pytz
from pytz import utc
from pandas._libs.tslibs.dtypes import NpyDatetimeUnit
from pandas._libs.tslibs.timezones import (
from pandas.compat import IS64
from pandas import (
import pandas._testing as tm
from pandas.tseries import offsets
from pandas.tseries.frequencies import to_offset
def test_utctimetuple():
    ts = Timestamp('2000-01-01', tz='UTC')
    result = ts.utctimetuple()
    expected = time.struct_time((2000, 1, 1, 0, 0, 0, 5, 1, 0))
    assert result == expected