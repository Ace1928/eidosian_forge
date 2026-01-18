import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
from pandas.api.types import is_integer
from pandas.core.arrays import IntegerArray
from pandas.core.arrays.integer import (
@pytest.mark.parametrize('a, b', [([1, None], [1, np.nan]), ([None], [np.nan]), ([None, np.nan], [np.nan, np.nan]), ([np.nan, np.nan], [np.nan, np.nan])])
def test_to_integer_array_none_is_nan(a, b):
    result = pd.array(a, dtype='Int64')
    expected = pd.array(b, dtype='Int64')
    tm.assert_extension_array_equal(result, expected)