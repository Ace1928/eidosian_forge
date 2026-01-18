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
def test_parse_date_time_multi_level_column_name(all_parsers):
    data = 'D,T,A,B\ndate, time,a,b\n2001-01-05, 09:00:00, 0.0, 10.\n2001-01-06, 00:00:00, 1.0, 11.\n'
    parser = all_parsers
    result = parser.read_csv_check_warnings(FutureWarning, "use 'date_format' instead", StringIO(data), header=[0, 1], parse_dates={'date_time': [0, 1]}, date_parser=pd.to_datetime)
    expected_data = [[datetime(2001, 1, 5, 9, 0, 0), 0.0, 10.0], [datetime(2001, 1, 6, 0, 0, 0), 1.0, 11.0]]
    expected = DataFrame(expected_data, columns=['date_time', ('A', 'a'), ('B', 'b')])
    tm.assert_frame_equal(result, expected)