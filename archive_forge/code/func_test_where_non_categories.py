import numpy as np
import pytest
from pandas.errors import InvalidIndexError
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_where_non_categories(self):
    ci = CategoricalIndex(['a', 'b', 'c', 'd'])
    mask = np.array([True, False, True, False])
    result = ci.where(mask, 2)
    expected = Index(['a', 2, 'c', 2], dtype=object)
    tm.assert_index_equal(result, expected)
    msg = 'Cannot setitem on a Categorical with a new category'
    with pytest.raises(TypeError, match=msg):
        ci._data._where(mask, 2)