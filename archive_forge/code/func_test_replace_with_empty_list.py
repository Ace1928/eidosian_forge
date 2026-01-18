from __future__ import annotations
from datetime import datetime
import re
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_replace_with_empty_list(self, frame_or_series):
    ser = Series([['a', 'b'], [], np.nan, [1]])
    obj = DataFrame({'col': ser})
    obj = tm.get_obj(obj, frame_or_series)
    expected = obj
    result = obj.replace([], np.nan)
    tm.assert_equal(result, expected)
    msg = 'NumPy boolean array indexing assignment cannot assign {size} input values to the 1 output values where the mask is true'
    with pytest.raises(ValueError, match=msg.format(size=0)):
        obj.replace({np.nan: []})
    with pytest.raises(ValueError, match=msg.format(size=2)):
        obj.replace({np.nan: ['dummy', 'alt']})