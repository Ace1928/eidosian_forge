import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_outer_sort_columns(self):
    df1 = DataFrame({'A': [0], 'B': [1], 0: 1})
    df2 = DataFrame({'A': [100]})
    result = concat([df1, df2], ignore_index=True, join='outer', sort=True)
    expected = DataFrame({0: [1.0, np.nan], 'A': [0, 100], 'B': [1.0, np.nan]})
    tm.assert_frame_equal(result, expected)