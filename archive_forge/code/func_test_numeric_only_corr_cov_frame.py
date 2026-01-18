import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('kernel', ['corr', 'cov'])
@pytest.mark.parametrize('use_arg', [True, False])
def test_numeric_only_corr_cov_frame(kernel, numeric_only, use_arg):
    df = DataFrame({'a': [1, 2, 3], 'b': 2, 'c': 3})
    df['c'] = df['c'].astype(object)
    arg = (df,) if use_arg else ()
    expanding = df.expanding()
    op = getattr(expanding, kernel)
    result = op(*arg, numeric_only=numeric_only)
    columns = ['a', 'b'] if numeric_only else ['a', 'b', 'c']
    df2 = df[columns].astype(float)
    arg2 = (df2,) if use_arg else ()
    expanding2 = df2.expanding()
    op2 = getattr(expanding2, kernel)
    expected = op2(*arg2, numeric_only=numeric_only)
    tm.assert_frame_equal(result, expected)