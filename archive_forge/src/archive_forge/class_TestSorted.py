import datetime
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
class TestSorted:
    """everything you wanted to test about sorting"""

    def test_sort_non_lexsorted(self):
        idx = MultiIndex([['A', 'B', 'C'], ['c', 'b', 'a']], [[0, 1, 2, 0, 1, 2], [0, 2, 1, 1, 0, 2]])
        df = DataFrame({'col': range(len(idx))}, index=idx, dtype='int64')
        assert df.index.is_monotonic_increasing is False
        sorted = df.sort_index()
        assert sorted.index.is_monotonic_increasing is True
        expected = DataFrame({'col': [1, 4, 5, 2]}, index=MultiIndex.from_tuples([('B', 'a'), ('B', 'c'), ('C', 'a'), ('C', 'b')]), dtype='int64')
        result = sorted.loc[pd.IndexSlice['B':'C', 'a':'c'], :]
        tm.assert_frame_equal(result, expected)