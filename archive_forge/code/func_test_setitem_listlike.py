import math
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
import pandas.core.common as com
def test_setitem_listlike(self):
    cat = Categorical(np.random.default_rng(2).integers(0, 5, size=150000).astype(np.int8)).add_categories([-1000])
    indexer = np.array([100000]).astype(np.int64)
    cat[indexer] = -1000
    result = cat.codes[np.array([100000]).astype(np.int64)]
    tm.assert_numpy_array_equal(result, np.array([5], dtype='int8'))