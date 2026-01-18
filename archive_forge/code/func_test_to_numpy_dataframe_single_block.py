import numpy as np
import pytest
from pandas.core.dtypes.dtypes import DatetimeTZDtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import (
from pandas.core.arrays.string_arrow import ArrowStringArrayNumpySemantics
@pytest.mark.parametrize('data, expected', [({'a': pd.array([1, 2, None])}, np.array([[1.0], [2.0], [np.nan]], dtype=float)), ({'a': [1, 2, 3], 'b': [1, 2, 3]}, np.array([[1, 1], [2, 2], [3, 3]], dtype=float))])
def test_to_numpy_dataframe_single_block(data, expected):
    df = pd.DataFrame(data)
    result = df.to_numpy(dtype=float, na_value=np.nan)
    tm.assert_numpy_array_equal(result, expected)