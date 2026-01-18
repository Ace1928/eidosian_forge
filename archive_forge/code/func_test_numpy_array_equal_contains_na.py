import copy
import numpy as np
import pytest
import pandas as pd
from pandas import Timestamp
import pandas._testing as tm
def test_numpy_array_equal_contains_na():
    a = np.array([True, False])
    b = np.array([True, pd.NA], dtype=object)
    msg = 'numpy array are different\n\nnumpy array values are different \\(50.0 %\\)\n\\[left\\]:  \\[True, False\\]\n\\[right\\]: \\[True, <NA>\\]'
    with pytest.raises(AssertionError, match=msg):
        tm.assert_numpy_array_equal(a, b)