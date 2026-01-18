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
def test_woy_boundary(self):
    d = datetime(2013, 12, 31)
    result = Timestamp(d).week
    expected = 1
    assert result == expected
    d = datetime(2008, 12, 28)
    result = Timestamp(d).week
    expected = 52
    assert result == expected
    d = datetime(2009, 12, 31)
    result = Timestamp(d).week
    expected = 53
    assert result == expected
    d = datetime(2010, 1, 1)
    result = Timestamp(d).week
    expected = 53
    assert result == expected
    d = datetime(2010, 1, 3)
    result = Timestamp(d).week
    expected = 53
    assert result == expected
    result = np.array([Timestamp(datetime(*args)).week for args in [(2000, 1, 1), (2000, 1, 2), (2005, 1, 1), (2005, 1, 2)]])
    assert (result == [52, 52, 53, 53]).all()