import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_numeric_only_frame(arithmetic_win_operators, numeric_only):
    kernel = arithmetic_win_operators
    df = DataFrame({'a': [1], 'b': 2, 'c': 3})
    df['c'] = df['c'].astype(object)
    expanding = df.expanding()
    op = getattr(expanding, kernel, None)
    if op is not None:
        result = op(numeric_only=numeric_only)
        columns = ['a', 'b'] if numeric_only else ['a', 'b', 'c']
        expected = df[columns].agg([kernel]).reset_index(drop=True).astype(float)
        assert list(expected.columns) == columns
        tm.assert_frame_equal(result, expected)