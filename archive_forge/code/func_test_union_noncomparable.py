from datetime import (
from hypothesis import (
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_union_noncomparable(self, sort):
    index = RangeIndex(start=0, stop=20, step=2)
    other = Index([datetime.now() + timedelta(i) for i in range(4)], dtype=object)
    result = index.union(other, sort=sort)
    expected = Index(np.concatenate((index, other)))
    tm.assert_index_equal(result, expected)
    result = other.union(index, sort=sort)
    expected = Index(np.concatenate((other, index)))
    tm.assert_index_equal(result, expected)