import re
import numpy as np
import pytest
from pandas.compat import PY311
from pandas import (
import pandas._testing as tm
from pandas.core.arrays.categorical import recode_for_categories
def test_set_categories_private(self):
    cat = Categorical(['a', 'b', 'c'], categories=['a', 'b', 'c', 'd'])
    cat._set_categories(['a', 'c', 'd', 'e'])
    expected = Categorical(['a', 'c', 'd'], categories=list('acde'))
    tm.assert_categorical_equal(cat, expected)
    cat = Categorical(['a', 'b', 'c'], categories=['a', 'b', 'c', 'd'])
    cat._set_categories(['a', 'c', 'd', 'e'], fastpath=True)
    expected = Categorical(['a', 'c', 'd'], categories=list('acde'))
    tm.assert_categorical_equal(cat, expected)