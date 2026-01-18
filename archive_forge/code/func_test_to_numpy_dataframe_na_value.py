import numpy as np
import pytest
from pandas.core.dtypes.dtypes import DatetimeTZDtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import (
from pandas.core.arrays.string_arrow import ArrowStringArrayNumpySemantics
@pytest.mark.parametrize('data', [{'a': [1, 2, 3], 'b': [1, 2, None]}, {'a': np.array([1, 2, 3]), 'b': np.array([1, 2, np.nan])}, {'a': pd.array([1, 2, 3]), 'b': pd.array([1, 2, None])}])
@pytest.mark.parametrize('dtype, na_value', [(float, np.nan), (object, None)])
def test_to_numpy_dataframe_na_value(data, dtype, na_value):
    df = pd.DataFrame(data)
    result = df.to_numpy(dtype=dtype, na_value=na_value)
    expected = np.array([[1, 1], [2, 2], [3, na_value]], dtype=dtype)
    tm.assert_numpy_array_equal(result, expected)