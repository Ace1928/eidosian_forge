import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_sort_columns_one_df(self):
    df1 = DataFrame({'A': [100], 0: 2})
    result = concat([df1], ignore_index=True, join='inner', sort=True)
    expected = DataFrame({0: [2], 'A': [100]})
    tm.assert_frame_equal(result, expected)