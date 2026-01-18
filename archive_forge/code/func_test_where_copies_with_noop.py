from datetime import datetime
from hypothesis import given
import numpy as np
import pytest
from pandas.core.dtypes.common import is_scalar
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas._testing._hypothesis import OPTIONAL_ONE_OF_ALL
def test_where_copies_with_noop(frame_or_series):
    result = frame_or_series([1, 2, 3, 4])
    expected = result.copy()
    col = result[0] if frame_or_series is DataFrame else result
    where_res = result.where(col < 5)
    where_res *= 2
    tm.assert_equal(result, expected)
    where_res = result.where(col > 5, [1, 2, 3, 4])
    where_res *= 2
    tm.assert_equal(result, expected)