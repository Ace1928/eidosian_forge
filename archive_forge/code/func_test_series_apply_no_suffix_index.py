import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.apply.common import series_transform_kernels
def test_series_apply_no_suffix_index(by_row):
    s = Series([4] * 3)
    result = s.apply(['sum', lambda x: x.sum(), lambda x: x.sum()], by_row=by_row)
    expected = Series([12, 12, 12], index=['sum', '<lambda>', '<lambda>'])
    tm.assert_series_equal(result, expected)