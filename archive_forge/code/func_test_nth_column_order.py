import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_nth_column_order():
    df = DataFrame([[1, 'b', 100], [1, 'a', 50], [1, 'a', np.nan], [2, 'c', 200], [2, 'd', 150]], columns=['A', 'C', 'B'])
    result = df.groupby('A').nth(0)
    expected = df.iloc[[0, 3]]
    tm.assert_frame_equal(result, expected)
    result = df.groupby('A').nth(-1, dropna='any')
    expected = df.iloc[[1, 4]]
    tm.assert_frame_equal(result, expected)