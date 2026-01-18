from itertools import permutations
import re
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
import pandas.core.common as com
def test_with_nans(self, closed):
    index = self.create_index(closed=closed)
    assert index.hasnans is False
    result = index.isna()
    expected = np.zeros(len(index), dtype=bool)
    tm.assert_numpy_array_equal(result, expected)
    result = index.notna()
    expected = np.ones(len(index), dtype=bool)
    tm.assert_numpy_array_equal(result, expected)
    index = self.create_index_with_nan(closed=closed)
    assert index.hasnans is True
    result = index.isna()
    expected = np.array([False, True] + [False] * (len(index) - 2))
    tm.assert_numpy_array_equal(result, expected)
    result = index.notna()
    expected = np.array([True, False] + [True] * (len(index) - 2))
    tm.assert_numpy_array_equal(result, expected)