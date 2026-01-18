from datetime import datetime
from io import StringIO
import numpy as np
import pytest
from pandas.errors import EmptyDataError
from pandas import (
import pandas._testing as tm
@xfail_pyarrow
def test_skip_rows_callable_not_in(all_parsers):
    parser = all_parsers
    data = '0,a\n1,b\n2,c\n3,d\n4,e'
    expected = DataFrame([[1, 'b'], [3, 'd']])
    result = parser.read_csv(StringIO(data), header=None, skiprows=lambda x: x not in [1, 3])
    tm.assert_frame_equal(result, expected)