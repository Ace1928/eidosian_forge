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
@pytest.mark.parametrize('date_str', ['2Q 2005', '2Q-200Y', '2Q-200', '22Q2005', '2Q200.', '6Q-20'])
def test_parsers_quarter_invalid(date_str):
    if date_str == '6Q-20':
        msg = f'Incorrect quarterly string is given, quarter must be between 1 and 4: {date_str}'
    else:
        msg = f'Unknown datetime string format, unable to parse: {date_str}'
    with pytest.raises(ValueError, match=msg):
        parsing.parse_datetime_string_with_reso(date_str)