import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_nth_indexed(slice_test_df, slice_test_grouped):
    result = slice_test_grouped.nth[0, 1, -2:]
    equivalent = slice_test_grouped.nth([0, 1, slice(-2, None)])
    expected = slice_test_df.iloc[[0, 1, 2, 3, 4, 6, 7]]
    tm.assert_frame_equal(result, expected)
    tm.assert_frame_equal(equivalent, expected)