import re
import numpy as np
import pytest
from pandas.compat import PY311
from pandas import (
import pandas._testing as tm
from pandas.core.arrays.categorical import recode_for_categories
def test_recode_to_categories_large(self):
    N = 1000
    codes = np.arange(N)
    old = Index(codes)
    expected = np.arange(N - 1, -1, -1, dtype=np.int16)
    new = Index(expected)
    result = recode_for_categories(codes, old, new)
    tm.assert_numpy_array_equal(result, expected)