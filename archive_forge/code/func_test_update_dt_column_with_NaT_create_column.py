import numpy as np
import pytest
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_update_dt_column_with_NaT_create_column(self):
    df = DataFrame({'A': [1, None], 'B': [pd.NaT, pd.to_datetime('2016-01-01')]})
    df2 = DataFrame({'A': [2, 3]})
    df.update(df2, overwrite=False)
    expected = DataFrame({'A': [1.0, 3.0], 'B': [pd.NaT, pd.to_datetime('2016-01-01')]})
    tm.assert_frame_equal(df, expected)