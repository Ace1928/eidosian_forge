import numpy as np
import pytest
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_update_with_different_dtype(self, using_copy_on_write):
    df = DataFrame({'a': [1, 3], 'b': [np.nan, 2]})
    df['c'] = np.nan
    with tm.assert_produces_warning(FutureWarning, match='incompatible dtype'):
        df.update({'c': Series(['foo'], index=[0])})
    expected = DataFrame({'a': [1, 3], 'b': [np.nan, 2], 'c': Series(['foo', np.nan], dtype='object')})
    tm.assert_frame_equal(df, expected)