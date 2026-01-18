from datetime import datetime
import numpy as np
import pytest
from pandas._libs import iNaT
import pandas._testing as tm
import pandas.core.algorithms as algos
def test_take_na_empty(self):
    result = algos.take(np.array([]), [-1, -1], allow_fill=True, fill_value=0.0)
    expected = np.array([0.0, 0.0])
    tm.assert_numpy_array_equal(result, expected)