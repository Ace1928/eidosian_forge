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
def test_start_end_fields(self, ts):
    assert ts.is_year_start
    assert ts.is_quarter_start
    assert ts.is_month_start
    assert not ts.is_year_end
    assert not ts.is_month_end
    assert not ts.is_month_end
    assert ts.is_year_start
    assert ts.is_quarter_start
    assert ts.is_month_start
    assert not ts.is_year_end
    assert not ts.is_month_end
    assert not ts.is_month_end