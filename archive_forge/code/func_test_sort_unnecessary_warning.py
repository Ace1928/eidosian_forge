import numpy as np
import pytest
from pandas.errors import (
from pandas import (
import pandas._testing as tm
from pandas.core.indexes.frozen import FrozenList
def test_sort_unnecessary_warning():
    midx = MultiIndex.from_tuples([(1.5, 2), (3.5, 3), (0, 1)])
    midx = midx.set_levels([2.5, np.nan, 1], level=0)
    result = midx.sort_values()
    expected = MultiIndex.from_tuples([(1, 3), (2.5, 1), (np.nan, 2)])
    tm.assert_index_equal(result, expected)