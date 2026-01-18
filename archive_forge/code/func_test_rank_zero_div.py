from datetime import datetime
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('input_key,input_value,output_value', [([1, 2], [1, 1], [1.0, 1.0]), ([1, 1, 2, 2], [1, 2, 1, 2], [0.5, 1.0, 0.5, 1.0]), ([1, 1, 2, 2], [1, 2, 1, np.nan], [0.5, 1.0, 1.0, np.nan]), ([1, 1, 2], [1, 2, np.nan], [0.5, 1.0, np.nan])])
def test_rank_zero_div(input_key, input_value, output_value):
    df = DataFrame({'A': input_key, 'B': input_value})
    result = df.groupby('A').rank(method='dense', pct=True)
    expected = DataFrame({'B': output_value})
    tm.assert_frame_equal(result, expected)