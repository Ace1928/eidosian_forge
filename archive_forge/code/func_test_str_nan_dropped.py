from io import StringIO
import numpy as np
import pytest
from pandas._libs.parsers import STR_NA_VALUES
from pandas import (
import pandas._testing as tm
@xfail_pyarrow
def test_str_nan_dropped(all_parsers):
    parser = all_parsers
    data = 'File: small.csv,,\n10010010233,0123,654\nfoo,,bar\n01001000155,4530,898'
    result = parser.read_csv(StringIO(data), header=None, names=['col1', 'col2', 'col3'], dtype={'col1': str, 'col2': str, 'col3': str}).dropna()
    expected = DataFrame({'col1': ['10010010233', '01001000155'], 'col2': ['0123', '4530'], 'col3': ['654', '898']}, index=[1, 3])
    tm.assert_frame_equal(result, expected)