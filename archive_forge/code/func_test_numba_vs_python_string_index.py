import numpy as np
import pytest
import pandas.util._test_decorators as td
from pandas import (
import pandas._testing as tm
def test_numba_vs_python_string_index():
    pytest.importorskip('pyarrow')
    df = DataFrame(1, index=Index(['a', 'b'], dtype='string[pyarrow_numpy]'), columns=Index(['x', 'y'], dtype='string[pyarrow_numpy]'))
    func = lambda x: x
    result = df.apply(func, engine='numba', axis=0)
    expected = df.apply(func, engine='python', axis=0)
    tm.assert_frame_equal(result, expected, check_column_type=False, check_index_type=False)