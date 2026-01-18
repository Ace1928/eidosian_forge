import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_sort_index_different_sortorder(self):
    A = np.arange(20).repeat(5)
    B = np.tile(np.arange(5), 20)
    indexer = np.random.default_rng(2).permutation(100)
    A = A.take(indexer)
    B = B.take(indexer)
    df = DataFrame({'A': A, 'B': B, 'C': np.random.default_rng(2).standard_normal(100)})
    ex_indexer = np.lexsort((df.B.max() - df.B, df.A))
    expected = df.take(ex_indexer)
    idf = df.set_index(['A', 'B'])
    result = idf.sort_index(ascending=[1, 0])
    expected = idf.take(ex_indexer)
    tm.assert_frame_equal(result, expected)
    result = idf['C'].sort_index(ascending=[1, 0])
    tm.assert_series_equal(result, expected['C'])