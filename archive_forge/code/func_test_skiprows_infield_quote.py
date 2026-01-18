from datetime import datetime
from io import StringIO
import numpy as np
import pytest
from pandas.errors import EmptyDataError
from pandas import (
import pandas._testing as tm
@xfail_pyarrow
def test_skiprows_infield_quote(all_parsers):
    parser = all_parsers
    data = 'a"\nb"\na\n1'
    expected = DataFrame({'a': [1]})
    result = parser.read_csv(StringIO(data), skiprows=2)
    tm.assert_frame_equal(result, expected)