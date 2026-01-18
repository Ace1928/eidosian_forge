from datetime import (
import numpy as np
import pytest
from pandas.errors import (
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import NumpyExtensionArray
from pandas.tests.arithmetic.common import (
def test_td64arr_floordiv_int(self, box_with_array):
    idx = TimedeltaIndex(np.arange(5, dtype='int64'))
    idx = tm.box_expected(idx, box_with_array)
    result = idx // 1
    tm.assert_equal(result, idx)
    pattern = 'floor_divide cannot use operands|Cannot divide int by Timedelta*'
    with pytest.raises(TypeError, match=pattern):
        1 // idx