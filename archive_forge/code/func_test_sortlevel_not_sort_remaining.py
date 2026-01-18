import numpy as np
import pytest
from pandas.errors import (
from pandas import (
import pandas._testing as tm
from pandas.core.indexes.frozen import FrozenList
def test_sortlevel_not_sort_remaining():
    mi = MultiIndex.from_tuples([[1, 1, 3], [1, 1, 1]], names=list('ABC'))
    sorted_idx, _ = mi.sortlevel('A', sort_remaining=False)
    assert sorted_idx.equals(mi)