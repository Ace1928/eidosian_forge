import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.util.version import Version
def test_sort_values_multicolumn(self):
    A = np.arange(5).repeat(20)
    B = np.tile(np.arange(5), 20)
    np.random.default_rng(2).shuffle(A)
    np.random.default_rng(2).shuffle(B)
    frame = DataFrame({'A': A, 'B': B, 'C': np.random.default_rng(2).standard_normal(100)})
    result = frame.sort_values(by=['A', 'B'])
    indexer = np.lexsort((frame['B'], frame['A']))
    expected = frame.take(indexer)
    tm.assert_frame_equal(result, expected)
    result = frame.sort_values(by=['A', 'B'], ascending=False)
    indexer = np.lexsort((frame['B'].rank(ascending=False), frame['A'].rank(ascending=False)))
    expected = frame.take(indexer)
    tm.assert_frame_equal(result, expected)
    result = frame.sort_values(by=['B', 'A'])
    indexer = np.lexsort((frame['A'], frame['B']))
    expected = frame.take(indexer)
    tm.assert_frame_equal(result, expected)