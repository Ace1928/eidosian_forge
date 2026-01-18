from datetime import (
from hypothesis import (
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_union_same_step_misaligned(self):
    left = RangeIndex(range(0, 20, 4))
    right = RangeIndex(range(1, 21, 4))
    result = left.union(right)
    expected = Index([0, 1, 4, 5, 8, 9, 12, 13, 16, 17])
    tm.assert_index_equal(result, expected, exact=True)