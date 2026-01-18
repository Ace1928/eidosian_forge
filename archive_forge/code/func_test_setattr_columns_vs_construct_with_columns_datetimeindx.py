import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_setattr_columns_vs_construct_with_columns_datetimeindx(self):
    idx = date_range('20130101', periods=4, freq='QE-NOV')
    df = DataFrame([[1, 1, 1, 5], [1, 1, 2, 5], [2, 1, 3, 5]], columns=['a', 'a', 'a', 'a'])
    df.columns = idx
    expected = DataFrame([[1, 1, 1, 5], [1, 1, 2, 5], [2, 1, 3, 5]], columns=idx)
    tm.assert_frame_equal(df, expected)