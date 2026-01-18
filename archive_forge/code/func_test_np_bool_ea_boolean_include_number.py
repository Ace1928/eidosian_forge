import numpy as np
import pytest
from pandas.core.dtypes.dtypes import ExtensionDtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import ExtensionArray
def test_np_bool_ea_boolean_include_number(self):
    df = DataFrame({'a': [1, 2, 3], 'b': pd.Series([True, False, True], dtype='boolean'), 'c': np.array([True, False, True]), 'd': pd.Categorical([True, False, True]), 'e': pd.arrays.SparseArray([True, False, True])})
    result = df.select_dtypes(include='number')
    expected = DataFrame({'a': [1, 2, 3]})
    tm.assert_frame_equal(result, expected)