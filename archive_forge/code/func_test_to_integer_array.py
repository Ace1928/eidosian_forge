import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
from pandas.api.types import is_integer
from pandas.core.arrays import IntegerArray
from pandas.core.arrays.integer import (
@pytest.mark.parametrize('values, to_dtype, result_dtype', [(np.array([1], dtype='int64'), None, Int64Dtype), (np.array([1, np.nan]), None, Int64Dtype), (np.array([1, np.nan]), 'int8', Int8Dtype)])
def test_to_integer_array(values, to_dtype, result_dtype):
    result = IntegerArray._from_sequence(values, dtype=to_dtype)
    assert result.dtype == result_dtype()
    expected = pd.array(values, dtype=result_dtype())
    tm.assert_extension_array_equal(result, expected)