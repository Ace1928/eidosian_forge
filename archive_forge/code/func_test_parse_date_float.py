from datetime import (
from io import StringIO
from dateutil.parser import parse as du_parse
import numpy as np
import pytest
import pytz
from pandas._libs.tslibs import parsing
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.indexes.datetimes import date_range
from pandas.core.tools.datetimes import start_caching_at
from pandas.io.parsers import read_csv
@pytest.mark.parametrize('data,expected', [('a\n135217135789158401\n1352171357E+5', DataFrame({'a': [135217135789158401, 135217135700000]}, dtype='float64')), ('a\n99999999999\n123456789012345\n1234E+0', DataFrame({'a': [99999999999, 123456789012345, 1234]}, dtype='float64'))])
@pytest.mark.parametrize('parse_dates', [True, False])
def test_parse_date_float(all_parsers, data, expected, parse_dates):
    parser = all_parsers
    result = parser.read_csv(StringIO(data), parse_dates=parse_dates)
    tm.assert_frame_equal(result, expected)