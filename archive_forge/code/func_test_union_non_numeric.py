from datetime import (
import numpy as np
import pytest
import pandas._testing as tm
from pandas.core.indexes.api import (
@pytest.mark.parametrize('dtype', ['f8', 'u8', 'i8'])
def test_union_non_numeric(self, dtype):
    index = Index(np.arange(5, dtype=dtype), dtype=dtype)
    assert index.dtype == dtype
    other = Index([datetime.now() + timedelta(i) for i in range(4)], dtype=object)
    result = index.union(other)
    expected = Index(np.concatenate((index, other)))
    tm.assert_index_equal(result, expected)
    result = other.union(index)
    expected = Index(np.concatenate((other, index)))
    tm.assert_index_equal(result, expected)