import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_take_when_index_has_step(self):
    idx = RangeIndex(1, 11, 3, name='foo')
    result = idx.take(np.array([1, 0, -1, -4]))
    expected = Index([4, 1, 10, 1], dtype=np.int64, name='foo')
    tm.assert_index_equal(result, expected)