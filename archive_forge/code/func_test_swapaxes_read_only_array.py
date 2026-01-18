import numpy as np
import pytest
from pandas.errors import SettingWithCopyWarning
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.copy_view.util import get_array
def test_swapaxes_read_only_array():
    df = DataFrame({'a': [1, 2], 'b': 3})
    msg = "'DataFrame.swapaxes' is deprecated"
    with tm.assert_produces_warning(FutureWarning, match=msg):
        df = df.swapaxes(axis1='index', axis2='columns')
    df.iloc[0, 0] = 100
    expected = DataFrame({0: [100, 3], 1: [2, 3]}, index=['a', 'b'])
    tm.assert_frame_equal(df, expected)