import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_pct_change_none_beginning_no_warning():
    df = DataFrame([[1, None], [2, 1], [3, 2], [4, 3], [5, 4]])
    result = df.pct_change()
    expected = DataFrame({0: [np.nan, 1, 0.5, 1 / 3, 0.25], 1: [np.nan, np.nan, 1, 0.5, 1 / 3]})
    tm.assert_frame_equal(result, expected)