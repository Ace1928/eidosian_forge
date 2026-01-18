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
def test_try_parse_dates():
    arr = np.array(['5/1/2000', '6/1/2000', '7/1/2000'], dtype=object)
    result = parsing.try_parse_dates(arr, parser=lambda x: du_parse(x, dayfirst=True))
    expected = np.array([du_parse(d, dayfirst=True) for d in arr])
    tm.assert_numpy_array_equal(result, expected)