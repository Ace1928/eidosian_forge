import calendar
from collections import deque
from datetime import (
from decimal import Decimal
import locale
from dateutil.parser import parse
from dateutil.tz.tz import tzoffset
import numpy as np
import pytest
import pytz
from pandas._libs import tslib
from pandas._libs.tslibs import (
from pandas.errors import (
import pandas.util._test_decorators as td
from pandas.core.dtypes.common import is_datetime64_ns_dtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import DatetimeArray
from pandas.core.tools import datetimes as tools
from pandas.core.tools.datetimes import start_caching_at
class TestDaysInMonth:

    @pytest.mark.parametrize('arg, format', [['2015-02-29', None], ['2015-02-29', '%Y-%m-%d'], ['2015-02-32', '%Y-%m-%d'], ['2015-04-31', '%Y-%m-%d']])
    def test_day_not_in_month_coerce(self, cache, arg, format):
        assert isna(to_datetime(arg, errors='coerce', format=format, cache=cache))

    def test_day_not_in_month_raise(self, cache):
        msg = 'day is out of range for month: 2015-02-29, at position 0'
        with pytest.raises(ValueError, match=msg):
            to_datetime('2015-02-29', errors='raise', cache=cache)

    @pytest.mark.parametrize('arg, format, msg', [('2015-02-29', '%Y-%m-%d', f'^day is out of range for month, at position 0. {PARSING_ERR_MSG}$'), ('2015-29-02', '%Y-%d-%m', f'^day is out of range for month, at position 0. {PARSING_ERR_MSG}$'), ('2015-02-32', '%Y-%m-%d', f'^unconverted data remains when parsing with format "%Y-%m-%d": "2", at position 0. {PARSING_ERR_MSG}$'), ('2015-32-02', '%Y-%d-%m', f"""^time data "2015-32-02" doesn't match format "%Y-%d-%m", at position 0. {PARSING_ERR_MSG}$"""), ('2015-04-31', '%Y-%m-%d', f'^day is out of range for month, at position 0. {PARSING_ERR_MSG}$'), ('2015-31-04', '%Y-%d-%m', f'^day is out of range for month, at position 0. {PARSING_ERR_MSG}$')])
    def test_day_not_in_month_raise_value(self, cache, arg, format, msg):
        with pytest.raises(ValueError, match=msg):
            to_datetime(arg, errors='raise', format=format, cache=cache)

    @pytest.mark.parametrize('expected, format', [['2015-02-29', None], ['2015-02-29', '%Y-%m-%d'], ['2015-02-29', '%Y-%m-%d'], ['2015-04-31', '%Y-%m-%d']])
    def test_day_not_in_month_ignore(self, cache, expected, format):
        result = to_datetime(expected, errors='ignore', format=format, cache=cache)
        assert result == expected