import numpy as np
import pytest
import pandas.util._test_decorators as td
from pandas import Series
import pandas._testing as tm
@pytest.mark.parametrize('index', [['a', 'b', 'c', 'd', 'e'], None])
def test_numpy_argwhere(index):
    s = Series(range(5), index=index, dtype=np.int64)
    result = np.argwhere(s > 2).astype(np.int64)
    expected = np.array([[3], [4]], dtype=np.int64)
    tm.assert_numpy_array_equal(result, expected)