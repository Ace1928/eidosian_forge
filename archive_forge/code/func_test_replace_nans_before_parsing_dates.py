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
@pytest.mark.parametrize(('key', 'value', 'warn'), [('date_parser', lambda x: pd.to_datetime(x, format='%Y-%m-%d'), FutureWarning), ('date_format', '%Y-%m-%d', None)])
def test_replace_nans_before_parsing_dates(all_parsers, key, value, warn):
    parser = all_parsers
    data = 'Test\n2012-10-01\n0\n2015-05-15\n#\n2017-09-09\n'
    result = parser.read_csv_check_warnings(warn, "use 'date_format' instead", StringIO(data), na_values={'Test': ['#', '0']}, parse_dates=['Test'], **{key: value})
    expected = DataFrame({'Test': [Timestamp('2012-10-01'), pd.NaT, Timestamp('2015-05-15'), pd.NaT, Timestamp('2017-09-09')]})
    tm.assert_frame_equal(result, expected)