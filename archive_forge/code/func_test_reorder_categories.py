import re
import numpy as np
import pytest
from pandas.compat import PY311
from pandas import (
import pandas._testing as tm
from pandas.core.arrays.categorical import recode_for_categories
def test_reorder_categories(self):
    cat = Categorical(['a', 'b', 'c', 'a'], ordered=True)
    old = cat.copy()
    new = Categorical(['a', 'b', 'c', 'a'], categories=['c', 'b', 'a'], ordered=True)
    res = cat.reorder_categories(['c', 'b', 'a'])
    tm.assert_categorical_equal(cat, old)
    tm.assert_categorical_equal(res, new)