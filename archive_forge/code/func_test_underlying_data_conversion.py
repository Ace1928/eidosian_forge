from datetime import timedelta
import re
import numpy as np
import pytest
from pandas.errors import IndexingError
from pandas import (
import pandas._testing as tm
def test_underlying_data_conversion(using_copy_on_write):
    df = DataFrame({c: [1, 2, 3] for c in ['a', 'b', 'c']})
    return_value = df.set_index(['a', 'b', 'c'], inplace=True)
    assert return_value is None
    s = Series([1], index=[(2, 2, 2)])
    df['val'] = 0
    df_original = df.copy()
    df
    if using_copy_on_write:
        with tm.raises_chained_assignment_error():
            df['val'].update(s)
        expected = df_original
    else:
        with tm.assert_produces_warning(FutureWarning, match='inplace method'):
            df['val'].update(s)
        expected = DataFrame({'a': [1, 2, 3], 'b': [1, 2, 3], 'c': [1, 2, 3], 'val': [0, 1, 0]})
        return_value = expected.set_index(['a', 'b', 'c'], inplace=True)
        assert return_value is None
    tm.assert_frame_equal(df, expected)