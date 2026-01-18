import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import IntervalArray
def test_set_na(self, left_right_dtypes):
    left, right = left_right_dtypes
    left = left.copy(deep=True)
    right = right.copy(deep=True)
    result = IntervalArray.from_arrays(left, right)
    if result.dtype.subtype.kind not in ['m', 'M']:
        msg = "'value' should be an interval type, got <.*NaTType'> instead."
        with pytest.raises(TypeError, match=msg):
            result[0] = pd.NaT
    if result.dtype.subtype.kind in ['i', 'u']:
        msg = 'Cannot set float NaN to integer-backed IntervalArray'
        with pytest.raises(TypeError, match=msg):
            result[0] = np.nan
        return
    result[0] = np.nan
    expected_left = Index([left._na_value] + list(left[1:]))
    expected_right = Index([right._na_value] + list(right[1:]))
    expected = IntervalArray.from_arrays(expected_left, expected_right)
    tm.assert_extension_array_equal(result, expected)