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
def test_timestamp_class_min_max_resolution():
    assert Timestamp.min == Timestamp(NaT._value + 1)
    assert Timestamp.min._creso == NpyDatetimeUnit.NPY_FR_ns.value
    assert Timestamp.max == Timestamp(np.iinfo(np.int64).max)
    assert Timestamp.max._creso == NpyDatetimeUnit.NPY_FR_ns.value
    assert Timestamp.resolution == Timedelta(1)
    assert Timestamp.resolution._creso == NpyDatetimeUnit.NPY_FR_ns.value