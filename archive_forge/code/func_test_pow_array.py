import operator
import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
from pandas.core import ops
from pandas.core.arrays import FloatingArray
def test_pow_array():
    a = pd.array([0, 0, 0, 1, 1, 1, None, None, None])
    b = pd.array([0, 1, None, 0, 1, None, 0, 1, None])
    result = a ** b
    expected = pd.array([1, 0, None, 1, 1, 1, 1, None, None])
    tm.assert_extension_array_equal(result, expected)