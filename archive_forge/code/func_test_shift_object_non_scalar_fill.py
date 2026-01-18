import numpy as np
import pytest
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_shift_object_non_scalar_fill(self):
    ser = Series(range(3))
    with pytest.raises(ValueError, match='fill_value must be a scalar'):
        ser.shift(1, fill_value=[])
    df = ser.to_frame()
    with pytest.raises(ValueError, match='fill_value must be a scalar'):
        df.shift(1, fill_value=np.arange(3))
    obj_ser = ser.astype(object)
    result = obj_ser.shift(1, fill_value={})
    assert result[0] == {}
    obj_df = obj_ser.to_frame()
    result = obj_df.shift(1, fill_value={})
    assert result.iloc[0, 0] == {}