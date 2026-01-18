import numpy as np
import pytest
from pandas.core.dtypes.dtypes import DatetimeTZDtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import (
from pandas.core.arrays.string_arrow import ArrowStringArrayNumpySemantics
def test_to_numpy_kwargs_raises():
    s = Series([1, 2, 3])
    msg = "to_numpy\\(\\) got an unexpected keyword argument 'foo'"
    with pytest.raises(TypeError, match=msg):
        s.to_numpy(foo=True)
    s = Series([1, 2, 3], dtype='Int64')
    with pytest.raises(TypeError, match=msg):
        s.to_numpy(foo=True)