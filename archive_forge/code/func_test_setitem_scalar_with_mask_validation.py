import operator
import numpy as np
import pytest
from pandas.compat.pyarrow import pa_version_under12p0
from pandas.core.dtypes.common import is_dtype_equal
import pandas as pd
import pandas._testing as tm
from pandas.core.arrays.string_arrow import (
def test_setitem_scalar_with_mask_validation(dtype):
    ser = pd.Series(['a', 'b', 'c'], dtype=dtype)
    mask = np.array([False, True, False])
    ser[mask] = None
    assert ser.array[1] is na_val(ser.dtype)
    ser = pd.Series(['a', 'b', 'c'], dtype=dtype)
    if type(ser.array) is pd.arrays.StringArray:
        msg = 'Cannot set non-string value'
    else:
        msg = 'Scalar must be NA or str'
    with pytest.raises(TypeError, match=msg):
        ser[mask] = 1