import numpy as np
import pytest
from pandas.core.dtypes.dtypes import CategoricalDtype
from pandas import (
import pandas._testing as tm
def test_set_dtype_same(self):
    c = Categorical(['a', 'b', 'c'])
    result = c._set_dtype(CategoricalDtype(['a', 'b', 'c']))
    tm.assert_categorical_equal(result, c)