from datetime import datetime
import struct
import numpy as np
import pytest
from pandas._libs import (
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import CategoricalDtype
import pandas as pd
from pandas import (
import pandas._testing as tm
import pandas.core.algorithms as algos
from pandas.core.arrays import (
import pandas.core.common as com
def test_uint64_overflow(self):
    exp = Series([2 ** 63], dtype=np.uint64)
    ser = Series([1, 2 ** 63, 2 ** 63], dtype=np.uint64)
    tm.assert_numpy_array_equal(algos.mode(ser.values), exp.values)
    tm.assert_series_equal(ser.mode(), exp)
    exp = Series([1, 2 ** 63], dtype=np.uint64)
    ser = Series([1, 2 ** 63], dtype=np.uint64)
    tm.assert_numpy_array_equal(algos.mode(ser.values), exp.values)
    tm.assert_series_equal(ser.mode(), exp)