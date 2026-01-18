import re
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_unbalanced(self):
    df = DataFrame({'A2010': [1.0, 2.0], 'A2011': [3.0, 4.0], 'B2010': [5.0, 6.0], 'X': ['X1', 'X2']})
    df['id'] = df.index
    exp_data = {'X': ['X1', 'X2', 'X1', 'X2'], 'A': [1.0, 2.0, 3.0, 4.0], 'B': [5.0, 6.0, np.nan, np.nan], 'id': [0, 1, 0, 1], 'year': [2010, 2010, 2011, 2011]}
    expected = DataFrame(exp_data)
    expected = expected.set_index(['id', 'year'])[['X', 'A', 'B']]
    result = wide_to_long(df, ['A', 'B'], i='id', j='year')
    tm.assert_frame_equal(result, expected)