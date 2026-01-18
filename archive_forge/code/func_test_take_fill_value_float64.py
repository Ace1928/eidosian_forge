import numpy as np
import pytest
from pandas.errors import InvalidIndexError
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import (
def test_take_fill_value_float64(self):
    idx = Index([1.0, 2.0, 3.0], name='xxx', dtype=np.float64)
    result = idx.take(np.array([1, 0, -1]))
    expected = Index([2.0, 1.0, 3.0], dtype=np.float64, name='xxx')
    tm.assert_index_equal(result, expected)
    result = idx.take(np.array([1, 0, -1]), fill_value=True)
    expected = Index([2.0, 1.0, np.nan], dtype=np.float64, name='xxx')
    tm.assert_index_equal(result, expected)
    result = idx.take(np.array([1, 0, -1]), allow_fill=False, fill_value=True)
    expected = Index([2.0, 1.0, 3.0], dtype=np.float64, name='xxx')
    tm.assert_index_equal(result, expected)
    msg = 'When allow_fill=True and fill_value is not None, all indices must be >= -1'
    with pytest.raises(ValueError, match=msg):
        idx.take(np.array([1, 0, -2]), fill_value=True)
    with pytest.raises(ValueError, match=msg):
        idx.take(np.array([1, 0, -5]), fill_value=True)
    msg = 'index -5 is out of bounds for (axis 0 with )?size 3'
    with pytest.raises(IndexError, match=msg):
        idx.take(np.array([1, -5]))