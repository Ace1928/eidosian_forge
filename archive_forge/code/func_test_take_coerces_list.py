from datetime import datetime
import numpy as np
import pytest
from pandas._libs import iNaT
import pandas._testing as tm
import pandas.core.algorithms as algos
def test_take_coerces_list(self):
    arr = [1, 2, 3]
    msg = 'take accepting non-standard inputs is deprecated'
    with tm.assert_produces_warning(FutureWarning, match=msg):
        result = algos.take(arr, [0, 0])
    expected = np.array([1, 1])
    tm.assert_numpy_array_equal(result, expected)