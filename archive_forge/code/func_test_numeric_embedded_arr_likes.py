import decimal
import numpy as np
from numpy import iinfo
import pytest
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('data,exp_data', [([[decimal.Decimal(3.14), 1.0], decimal.Decimal(1.6), 0.1], [[3.14, 1.0], 1.6, 0.1]), ([np.array([decimal.Decimal(3.14), 1.0]), 0.1], [[3.14, 1.0], 0.1])])
def test_numeric_embedded_arr_likes(data, exp_data):
    df = DataFrame({'a': data})
    df['a'] = df['a'].apply(to_numeric)
    expected = DataFrame({'a': exp_data})
    tm.assert_frame_equal(df, expected)