import re
import numpy as np
import pytest
from pandas.compat import PY311
from pandas import (
import pandas._testing as tm
from pandas.core.arrays.categorical import recode_for_categories
def test_set_categories(self):
    cat = Categorical(['a', 'b', 'c', 'a'], ordered=True)
    exp_categories = Index(['c', 'b', 'a'])
    exp_values = np.array(['a', 'b', 'c', 'a'], dtype=np.object_)
    cat = cat.set_categories(['c', 'b', 'a'])
    res = cat.set_categories(['a', 'b', 'c'])
    tm.assert_index_equal(cat.categories, exp_categories)
    tm.assert_numpy_array_equal(cat.__array__(), exp_values)
    exp_categories_back = Index(['a', 'b', 'c'])
    tm.assert_index_equal(res.categories, exp_categories_back)
    tm.assert_numpy_array_equal(res.__array__(), exp_values)
    cat = Categorical(['a', 'b', 'c', 'a'], ordered=True)
    res = cat.set_categories(['a'])
    tm.assert_numpy_array_equal(res.codes, np.array([0, -1, -1, 0], dtype=np.int8))
    res = cat.set_categories(['a', 'b', 'd'])
    tm.assert_numpy_array_equal(res.codes, np.array([0, 1, -1, 0], dtype=np.int8))
    tm.assert_index_equal(res.categories, Index(['a', 'b', 'd']))
    cat = cat.set_categories(['a', 'b', 'c', 'd'])
    exp_categories = Index(['a', 'b', 'c', 'd'])
    tm.assert_index_equal(cat.categories, exp_categories)
    c = Categorical([1, 2, 3, 4, 1], categories=[1, 2, 3, 4], ordered=True)
    tm.assert_numpy_array_equal(c._codes, np.array([0, 1, 2, 3, 0], dtype=np.int8))
    tm.assert_index_equal(c.categories, Index([1, 2, 3, 4]))
    exp = np.array([1, 2, 3, 4, 1], dtype=np.int64)
    tm.assert_numpy_array_equal(np.asarray(c), exp)
    c = c.set_categories([4, 3, 2, 1])
    tm.assert_numpy_array_equal(c._codes, np.array([3, 2, 1, 0, 3], dtype=np.int8))
    tm.assert_index_equal(c.categories, Index([4, 3, 2, 1]))
    exp = np.array([1, 2, 3, 4, 1], dtype=np.int64)
    tm.assert_numpy_array_equal(np.asarray(c), exp)
    assert c.min() == 4
    assert c.max() == 1
    c2 = c.set_categories([4, 3, 2, 1], ordered=False)
    assert not c2.ordered
    tm.assert_numpy_array_equal(np.asarray(c), np.asarray(c2))
    c2 = c.set_ordered(False).set_categories([4, 3, 2, 1])
    assert not c2.ordered
    tm.assert_numpy_array_equal(np.asarray(c), np.asarray(c2))