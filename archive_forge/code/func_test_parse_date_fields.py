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
def test_parse_date_fields(all_parsers):
    parser = all_parsers
    data = 'year,month,day,a\n2001,01,10,10.\n2001,02,1,11.'
    result = parser.read_csv_check_warnings(FutureWarning, "use 'date_format' instead", StringIO(data), header=0, parse_dates={'ymd': [0, 1, 2]}, date_parser=lambda x: x, raise_on_extra_warnings=False)
    expected = DataFrame([[datetime(2001, 1, 10), 10.0], [datetime(2001, 2, 1), 11.0]], columns=['ymd', 'a'])
    tm.assert_frame_equal(result, expected)