import re
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_wide_to_long_pyarrow_string_columns():
    pytest.importorskip('pyarrow')
    df = DataFrame({'ID': {0: 1}, 'R_test1': {0: 1}, 'R_test2': {0: 1}, 'R_test3': {0: 2}, 'D': {0: 1}})
    df.columns = df.columns.astype('string[pyarrow_numpy]')
    result = wide_to_long(df, stubnames='R', i='ID', j='UNPIVOTED', sep='_', suffix='.*')
    expected = DataFrame([[1, 1], [1, 1], [1, 2]], columns=Index(['D', 'R'], dtype=object), index=pd.MultiIndex.from_arrays([[1, 1, 1], Index(['test1', 'test2', 'test3'], dtype='string[pyarrow_numpy]')], names=['ID', 'UNPIVOTED']))
    tm.assert_frame_equal(result, expected)