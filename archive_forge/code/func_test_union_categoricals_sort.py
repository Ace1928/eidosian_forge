import numpy as np
import pytest
from pandas.core.dtypes.concat import union_categoricals
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_union_categoricals_sort(self):
    c1 = Categorical(['x', 'y', 'z'])
    c2 = Categorical(['a', 'b', 'c'])
    result = union_categoricals([c1, c2], sort_categories=True)
    expected = Categorical(['x', 'y', 'z', 'a', 'b', 'c'], categories=['a', 'b', 'c', 'x', 'y', 'z'])
    tm.assert_categorical_equal(result, expected)
    c1 = Categorical(['a', 'b'], categories=['b', 'a', 'c'])
    c2 = Categorical(['b', 'c'], categories=['b', 'a', 'c'])
    result = union_categoricals([c1, c2], sort_categories=True)
    expected = Categorical(['a', 'b', 'b', 'c'], categories=['a', 'b', 'c'])
    tm.assert_categorical_equal(result, expected)
    c1 = Categorical(['a', 'b'], categories=['c', 'a', 'b'])
    c2 = Categorical(['b', 'c'], categories=['c', 'a', 'b'])
    result = union_categoricals([c1, c2], sort_categories=True)
    expected = Categorical(['a', 'b', 'b', 'c'], categories=['a', 'b', 'c'])
    tm.assert_categorical_equal(result, expected)
    c1 = Categorical(['a', 'b'], categories=['a', 'b', 'c'])
    c2 = Categorical(['b', 'c'], categories=['a', 'b', 'c'])
    result = union_categoricals([c1, c2], sort_categories=True)
    expected = Categorical(['a', 'b', 'b', 'c'], categories=['a', 'b', 'c'])
    tm.assert_categorical_equal(result, expected)
    c1 = Categorical(['x', np.nan])
    c2 = Categorical([np.nan, 'b'])
    result = union_categoricals([c1, c2], sort_categories=True)
    expected = Categorical(['x', np.nan, np.nan, 'b'], categories=['b', 'x'])
    tm.assert_categorical_equal(result, expected)
    c1 = Categorical([np.nan])
    c2 = Categorical([np.nan])
    result = union_categoricals([c1, c2], sort_categories=True)
    expected = Categorical([np.nan, np.nan])
    tm.assert_categorical_equal(result, expected)
    c1 = Categorical([])
    c2 = Categorical([])
    result = union_categoricals([c1, c2], sort_categories=True)
    expected = Categorical([])
    tm.assert_categorical_equal(result, expected)
    c1 = Categorical(['b', 'a'], categories=['b', 'a', 'c'], ordered=True)
    c2 = Categorical(['a', 'c'], categories=['b', 'a', 'c'], ordered=True)
    msg = 'Cannot use sort_categories=True with ordered Categoricals'
    with pytest.raises(TypeError, match=msg):
        union_categoricals([c1, c2], sort_categories=True)