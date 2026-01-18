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
def test_read_csv_with_custom_date_parser_parse_dates_false(all_parsers):

    def __custom_date_parser(time):
        time = time.astype(np.float64)
        time = time.astype(int)
        return pd.to_timedelta(time, unit='s')
    testdata = StringIO('time e\n        41047.00 -93.77\n        41048.00 -95.79\n        41049.00 -98.73\n        41050.00 -93.99\n        41051.00 -97.72\n        ')
    result = all_parsers.read_csv_check_warnings(FutureWarning, "Please use 'date_format' instead", testdata, delim_whitespace=True, parse_dates=False, date_parser=__custom_date_parser, index_col='time')
    time = Series([41047.0, 41048.0, 41049.0, 41050.0, 41051.0], name='time')
    expected = DataFrame({'e': [-93.77, -95.79, -98.73, -93.99, -97.72]}, index=time)
    tm.assert_frame_equal(result, expected)