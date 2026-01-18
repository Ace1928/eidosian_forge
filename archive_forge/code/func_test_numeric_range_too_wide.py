from io import StringIO
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
@xfail_pyarrow
@pytest.mark.parametrize('exp_data', [[str(-1), str(2 ** 63)], [str(2 ** 63), str(-1)]])
def test_numeric_range_too_wide(all_parsers, exp_data):
    parser = all_parsers
    data = '\n'.join(exp_data)
    expected = DataFrame(exp_data)
    result = parser.read_csv(StringIO(data), header=None)
    tm.assert_frame_equal(result, expected)