import numpy as np
import pytest
from pandas import Categorical
import pandas._testing as tm
def test_take_allow_fill(self):
    cat = Categorical(['a', 'a', 'b'])
    result = cat.take([0, -1, -1], allow_fill=True)
    expected = Categorical(['a', np.nan, np.nan], categories=['a', 'b'])
    tm.assert_categorical_equal(result, expected)