import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
from pandas.api.types import is_integer
from pandas.core.arrays import IntegerArray
from pandas.core.arrays.integer import (
def test_to_integer_array_float():
    result = IntegerArray._from_sequence([1.0, 2.0], dtype='Int64')
    expected = pd.array([1, 2], dtype='Int64')
    tm.assert_extension_array_equal(result, expected)
    with pytest.raises(TypeError, match='cannot safely cast non-equivalent'):
        IntegerArray._from_sequence([1.5, 2.0], dtype='Int64')
    result = IntegerArray._from_sequence(np.array([1.0, 2.0], dtype='float32'), dtype='Int64')
    assert result.dtype == Int64Dtype()