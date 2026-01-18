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
def test_parse_tz_aware(all_parsers):
    parser = all_parsers
    data = 'Date,x\n2012-06-13T01:39:00Z,0.5'
    result = parser.read_csv(StringIO(data), index_col=0, parse_dates=True)
    if parser.engine == 'pyarrow':
        result.index = result.index.as_unit('ns')
    expected = DataFrame({'x': [0.5]}, index=Index([Timestamp('2012-06-13 01:39:00+00:00')], name='Date'))
    if parser.engine == 'pyarrow':
        expected_tz = pytz.utc
    else:
        expected_tz = timezone.utc
    tm.assert_frame_equal(result, expected)
    assert result.index.tz is expected_tz