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
def test_read_csv_with_custom_date_parser(all_parsers):

    def __custom_date_parser(time):
        time = time.astype(np.float64)
        time = time.astype(int)
        return pd.to_timedelta(time, unit='s')
    testdata = StringIO('time e n h\n        41047.00 -98573.7297 871458.0640 389.0089\n        41048.00 -98573.7299 871458.0640 389.0089\n        41049.00 -98573.7300 871458.0642 389.0088\n        41050.00 -98573.7299 871458.0643 389.0088\n        41051.00 -98573.7302 871458.0640 389.0086\n        ')
    result = all_parsers.read_csv_check_warnings(FutureWarning, "Please use 'date_format' instead", testdata, delim_whitespace=True, parse_dates=True, date_parser=__custom_date_parser, index_col='time')
    time = [41047, 41048, 41049, 41050, 41051]
    time = pd.TimedeltaIndex([pd.to_timedelta(i, unit='s') for i in time], name='time')
    expected = DataFrame({'e': [-98573.7297, -98573.7299, -98573.73, -98573.7299, -98573.7302], 'n': [871458.064, 871458.064, 871458.0642, 871458.0643, 871458.064], 'h': [389.0089, 389.0089, 389.0088, 389.0088, 389.0086]}, index=time)
    tm.assert_frame_equal(result, expected)