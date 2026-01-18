import re
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
import pandas as pd
import pandas._testing as tm
from pandas.core.arrays import IntervalArray
@pytest.mark.parametrize('frame', [False, True])
def test_replace_empty_copy(self, frame):
    obj = pd.Series([], dtype=np.float64)
    if frame:
        obj = obj.to_frame()
    res = obj.replace(4, 5, inplace=True)
    assert res is None
    res = obj.replace(4, 5, inplace=False)
    tm.assert_equal(res, obj)
    assert res is not obj