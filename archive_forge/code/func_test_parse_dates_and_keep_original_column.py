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
def test_parse_dates_and_keep_original_column(all_parsers):
    parser = all_parsers
    data = 'A\n20150908\n20150909\n'
    depr_msg = "The 'keep_date_col' keyword in pd.read_csv is deprecated"
    with tm.assert_produces_warning(FutureWarning, match=depr_msg, check_stacklevel=False):
        result = parser.read_csv(StringIO(data), parse_dates={'date': ['A']}, keep_date_col=True)
    expected_data = [Timestamp('2015-09-08'), Timestamp('2015-09-09')]
    expected = DataFrame({'date': expected_data, 'A': expected_data})
    tm.assert_frame_equal(result, expected)