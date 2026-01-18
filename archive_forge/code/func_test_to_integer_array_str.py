import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
from pandas.api.types import is_integer
from pandas.core.arrays import IntegerArray
from pandas.core.arrays.integer import (
def test_to_integer_array_str():
    result = IntegerArray._from_sequence(['1', '2', None], dtype='Int64')
    expected = pd.array([1, 2, np.nan], dtype='Int64')
    tm.assert_extension_array_equal(result, expected)
    with pytest.raises(ValueError, match='invalid literal for int\\(\\) with base 10: .*'):
        IntegerArray._from_sequence(['1', '2', ''], dtype='Int64')
    with pytest.raises(ValueError, match='invalid literal for int\\(\\) with base 10: .*'):
        IntegerArray._from_sequence(['1.5', '2.0'], dtype='Int64')