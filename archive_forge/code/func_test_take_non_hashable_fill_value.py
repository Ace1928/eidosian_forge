from datetime import datetime
import numpy as np
import pytest
from pandas._libs import iNaT
import pandas._testing as tm
import pandas.core.algorithms as algos
def test_take_non_hashable_fill_value(self):
    arr = np.array([1, 2, 3])
    indexer = np.array([1, -1])
    with pytest.raises(ValueError, match='fill_value must be a scalar'):
        algos.take(arr, indexer, allow_fill=True, fill_value=[1])
    arr = np.array([1, 2, 3], dtype=object)
    result = algos.take(arr, indexer, allow_fill=True, fill_value=[1])
    expected = np.array([2, [1]], dtype=object)
    tm.assert_numpy_array_equal(result, expected)