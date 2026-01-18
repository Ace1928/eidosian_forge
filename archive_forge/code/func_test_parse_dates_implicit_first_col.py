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
@skip_pyarrow
def test_parse_dates_implicit_first_col(all_parsers):
    data = 'A,B,C\n20090101,a,1,2\n20090102,b,3,4\n20090103,c,4,5\n'
    parser = all_parsers
    result = parser.read_csv(StringIO(data), parse_dates=True)
    expected = parser.read_csv(StringIO(data), index_col=0, parse_dates=True)
    tm.assert_frame_equal(result, expected)