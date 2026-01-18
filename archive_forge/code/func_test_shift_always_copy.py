import numpy as np
import pytest
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('obj', [Series([np.arange(5)]), date_range('1/1/2011', periods=24, freq='h'), Series(range(5), index=date_range('2017', periods=5))])
@pytest.mark.parametrize('shift_size', [0, 1, 2])
def test_shift_always_copy(self, obj, shift_size, frame_or_series):
    if frame_or_series is not Series:
        obj = obj.to_frame()
    assert obj.shift(shift_size) is not obj