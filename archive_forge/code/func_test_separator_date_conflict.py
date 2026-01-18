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
@xfail_pyarrow
def test_separator_date_conflict(all_parsers):
    parser = all_parsers
    data = '06-02-2013;13:00;1-000.215'
    expected = DataFrame([[datetime(2013, 6, 2, 13, 0, 0), 1000.215]], columns=['Date', 2])
    depr_msg = "Support for nested sequences for 'parse_dates' in pd.read_csv is deprecated"
    with tm.assert_produces_warning(FutureWarning, match=depr_msg, check_stacklevel=False):
        df = parser.read_csv(StringIO(data), sep=';', thousands='-', parse_dates={'Date': [0, 1]}, header=None)
    tm.assert_frame_equal(df, expected)