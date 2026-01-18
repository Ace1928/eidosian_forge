import numpy as np
import pytest
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_shift_categorical_fill_value(self, frame_or_series):
    ts = frame_or_series(['a', 'b', 'c', 'd'], dtype='category')
    res = ts.shift(1, fill_value='a')
    expected = frame_or_series(pd.Categorical(['a', 'a', 'b', 'c'], categories=['a', 'b', 'c', 'd'], ordered=False))
    tm.assert_equal(res, expected)
    msg = 'Cannot setitem on a Categorical with a new category \\(f\\)'
    with pytest.raises(TypeError, match=msg):
        ts.shift(1, fill_value='f')