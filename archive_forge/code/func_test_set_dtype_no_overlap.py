import numpy as np
import pytest
from pandas.core.dtypes.dtypes import CategoricalDtype
from pandas import (
import pandas._testing as tm
def test_set_dtype_no_overlap(self):
    c = Categorical(['a', 'b', 'c'], ['d', 'e'])
    result = c._set_dtype(CategoricalDtype(['a', 'b']))
    expected = Categorical([None, None, None], categories=['a', 'b'])
    tm.assert_categorical_equal(result, expected)