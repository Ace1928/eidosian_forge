import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
from pandas.arrays import BooleanArray
from pandas.core.arrays.boolean import coerce_to_array
@pytest.mark.parametrize('a, b', [([True, False, None, np.nan, pd.NA], [True, False, None, None, None]), ([True, np.nan], [True, None]), ([True, pd.NA], [True, None]), ([np.nan, np.nan], [None, None]), (np.array([np.nan, np.nan], dtype=float), [None, None])])
def test_to_boolean_array_missing_indicators(a, b):
    result = pd.array(a, dtype='boolean')
    expected = pd.array(b, dtype='boolean')
    tm.assert_extension_array_equal(result, expected)