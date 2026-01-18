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
def test_parse_dot_separated_dates(all_parsers):
    parser = all_parsers
    data = 'a,b\n27.03.2003 14:55:00.000,1\n03.08.2003 15:20:00.000,2'
    if parser.engine == 'pyarrow':
        expected_index = Index(['27.03.2003 14:55:00.000', '03.08.2003 15:20:00.000'], dtype='object', name='a')
        warn = None
    else:
        expected_index = DatetimeIndex(['2003-03-27 14:55:00', '2003-08-03 15:20:00'], dtype='datetime64[ns]', name='a')
        warn = UserWarning
    msg = 'when dayfirst=False \\(the default\\) was specified'
    result = parser.read_csv_check_warnings(warn, msg, StringIO(data), parse_dates=True, index_col=0, raise_on_extra_warnings=False)
    expected = DataFrame({'b': [1, 2]}, index=expected_index)
    tm.assert_frame_equal(result, expected)