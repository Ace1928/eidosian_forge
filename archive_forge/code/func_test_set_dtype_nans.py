import collections
import numpy as np
import pytest
from pandas.core.dtypes.dtypes import CategoricalDtype
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_set_dtype_nans(self):
    c = Categorical(['a', 'b', np.nan])
    result = c._set_dtype(CategoricalDtype(['a', 'c']))
    tm.assert_numpy_array_equal(result.codes, np.array([0, -1, -1], dtype='int8'))