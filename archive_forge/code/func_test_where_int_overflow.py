from datetime import datetime
from hypothesis import given
import numpy as np
import pytest
from pandas.core.dtypes.common import is_scalar
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas._testing._hypothesis import OPTIONAL_ONE_OF_ALL
@pytest.mark.parametrize('replacement', [0.001, True, 'snake', None, datetime(2022, 5, 4)])
def test_where_int_overflow(replacement, using_infer_string, request):
    df = DataFrame([[1.0, 2e+25, 'nine'], [np.nan, 0.1, None]])
    if using_infer_string and replacement not in (None, 'snake'):
        request.node.add_marker(pytest.mark.xfail(reason="Can't set non-string into string column"))
    result = df.where(pd.notnull(df), replacement)
    expected = DataFrame([[1.0, 2e+25, 'nine'], [replacement, 0.1, replacement]])
    tm.assert_frame_equal(result, expected)