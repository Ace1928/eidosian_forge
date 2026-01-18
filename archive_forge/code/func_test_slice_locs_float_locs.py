import numpy as np
import pytest
from pandas.errors import InvalidIndexError
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import (
@pytest.mark.parametrize('dtype', [int, float])
def test_slice_locs_float_locs(self, dtype):
    index = Index(np.array([0, 1, 2, 5, 6, 7, 9, 10], dtype=dtype))
    n = len(index)
    assert index.slice_locs(5.0, 10.0) == (3, n)
    assert index.slice_locs(4.5, 10.5) == (3, 8)
    index2 = index[::-1]
    assert index2.slice_locs(8.5, 1.5) == (2, 6)
    assert index2.slice_locs(10.5, -1) == (0, n)