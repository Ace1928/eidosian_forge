import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
from pandas.api.indexers import check_array_indexer
@pytest.mark.parametrize('indexer, expected', [([1, 2], np.array([1, 2], dtype=np.intp)), (np.array([1, 2], dtype='int64'), np.array([1, 2], dtype=np.intp)), (pd.array([1, 2], dtype='Int32'), np.array([1, 2], dtype=np.intp)), (pd.Index([1, 2]), np.array([1, 2], dtype=np.intp)), ([True, False, True], np.array([True, False, True], dtype=np.bool_)), (np.array([True, False, True]), np.array([True, False, True], dtype=np.bool_)), (pd.array([True, False, True], dtype='boolean'), np.array([True, False, True], dtype=np.bool_)), ([], np.array([], dtype=np.intp))])
def test_valid_input(indexer, expected):
    arr = np.array([1, 2, 3])
    result = check_array_indexer(arr, indexer)
    tm.assert_numpy_array_equal(result, expected)