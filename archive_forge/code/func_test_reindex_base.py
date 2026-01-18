import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_reindex_base(idx):
    expected = np.arange(idx.size, dtype=np.intp)
    actual = idx.get_indexer(idx)
    tm.assert_numpy_array_equal(expected, actual)
    with pytest.raises(ValueError, match='Invalid fill method'):
        idx.get_indexer(idx, method='invalid')