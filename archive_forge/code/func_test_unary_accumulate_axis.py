from functools import partial
import re
import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
from pandas.api.types import is_extension_array_dtype
def test_unary_accumulate_axis():
    df = pd.DataFrame({'a': [1, 3, 2, 4]})
    result = np.maximum.accumulate(df)
    expected = pd.DataFrame({'a': [1, 3, 3, 4]})
    tm.assert_frame_equal(result, expected)
    df = pd.DataFrame({'a': [1, 3, 2, 4], 'b': [0.1, 4.0, 3.0, 2.0]})
    result = np.maximum.accumulate(df)
    expected = pd.DataFrame({'a': [1.0, 3.0, 3.0, 4.0], 'b': [0.1, 4.0, 4.0, 4.0]})
    tm.assert_frame_equal(result, expected)
    result = np.maximum.accumulate(df, axis=0)
    tm.assert_frame_equal(result, expected)
    result = np.maximum.accumulate(df, axis=1)
    expected = pd.DataFrame({'a': [1.0, 3.0, 2.0, 4.0], 'b': [1.0, 4.0, 3.0, 4.0]})
    tm.assert_frame_equal(result, expected)