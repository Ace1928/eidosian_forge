import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('values', [[['a'], ['x']], [[], []]])
def test_reindex_empty_with_level(values):
    idx = MultiIndex.from_arrays(values)
    result, result_indexer = idx.reindex(np.array(['b']), level=0)
    expected = MultiIndex(levels=[['b'], values[1]], codes=[[], []])
    expected_indexer = np.array([], dtype=result_indexer.dtype)
    tm.assert_index_equal(result, expected)
    tm.assert_numpy_array_equal(result_indexer, expected_indexer)