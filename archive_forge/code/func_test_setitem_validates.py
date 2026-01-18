import operator
import numpy as np
import pytest
from pandas.compat.pyarrow import pa_version_under12p0
from pandas.core.dtypes.common import is_dtype_equal
import pandas as pd
import pandas._testing as tm
from pandas.core.arrays.string_arrow import (
def test_setitem_validates(cls, dtype):
    arr = cls._from_sequence(['a', 'b'], dtype=dtype)
    if cls is pd.arrays.StringArray:
        msg = "Cannot set non-string value '10' into a StringArray."
    else:
        msg = 'Scalar must be NA or str'
    with pytest.raises(TypeError, match=msg):
        arr[0] = 10
    if cls is pd.arrays.StringArray:
        msg = 'Must provide strings.'
    else:
        msg = 'Scalar must be NA or str'
    with pytest.raises(TypeError, match=msg):
        arr[:] = np.array([1, 2])