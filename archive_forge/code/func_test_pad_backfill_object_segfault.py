from datetime import datetime
from itertools import permutations
import numpy as np
from pandas._libs import algos as libalgos
import pandas._testing as tm
def test_pad_backfill_object_segfault(self):
    old = np.array([], dtype='O')
    new = np.array([datetime(2010, 12, 31)], dtype='O')
    result = libalgos.pad['object'](old, new)
    expected = np.array([-1], dtype=np.intp)
    tm.assert_numpy_array_equal(result, expected)
    result = libalgos.pad['object'](new, old)
    expected = np.array([], dtype=np.intp)
    tm.assert_numpy_array_equal(result, expected)
    result = libalgos.backfill['object'](old, new)
    expected = np.array([-1], dtype=np.intp)
    tm.assert_numpy_array_equal(result, expected)
    result = libalgos.backfill['object'](new, old)
    expected = np.array([], dtype=np.intp)
    tm.assert_numpy_array_equal(result, expected)