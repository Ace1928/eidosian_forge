from datetime import datetime
import re
from dateutil.parser import parse as du_parse
from dateutil.tz import tzlocal
from hypothesis import given
import numpy as np
import pytest
from pandas._libs.tslibs import (
from pandas._libs.tslibs.parsing import parse_datetime_string_with_reso
from pandas.compat import (
import pandas.util._test_decorators as td
import pandas._testing as tm
from pandas._testing._hypothesis import DATETIME_NO_TZ
@pytest.mark.parametrize('date_str,freq,expected', [('2013Q2', None, datetime(2013, 4, 1)), ('2013Q2', 'Y-APR', datetime(2012, 8, 1)), ('2013-Q2', 'Y-DEC', datetime(2013, 4, 1))])
def test_parsers_quarterly_with_freq(date_str, freq, expected):
    result, _ = parsing.parse_datetime_string_with_reso(date_str, freq=freq)
    assert result == expected