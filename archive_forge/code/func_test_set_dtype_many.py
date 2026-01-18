import numpy as np
import pytest
from pandas.core.dtypes.dtypes import CategoricalDtype
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('values, categories, new_categories', [(['a', 'b', 'a'], ['a', 'b'], ['a', 'b']), (['a', 'b', 'a'], ['a', 'b'], ['b', 'a']), (['b', 'a', 'a'], ['a', 'b'], ['a', 'b']), (['b', 'a', 'a'], ['a', 'b'], ['b', 'a']), (['a', 'b', 'c'], ['a', 'b'], ['a', 'b']), (['a', 'b', 'c'], ['a', 'b'], ['b', 'a']), (['b', 'a', 'c'], ['a', 'b'], ['a', 'b']), (['b', 'a', 'c'], ['a', 'b'], ['a', 'b']), (['a', 'b', 'c'], ['a', 'b'], ['a']), (['a', 'b', 'c'], ['a', 'b'], ['b']), (['b', 'a', 'c'], ['a', 'b'], ['a']), (['b', 'a', 'c'], ['a', 'b'], ['a']), (['a', 'b', 'c'], ['a', 'b'], ['d', 'e'])])
@pytest.mark.parametrize('ordered', [True, False])
def test_set_dtype_many(self, values, categories, new_categories, ordered):
    c = Categorical(values, categories)
    expected = Categorical(values, new_categories, ordered)
    result = c._set_dtype(expected.dtype)
    tm.assert_categorical_equal(result, expected)