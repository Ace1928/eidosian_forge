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
def test_numeric_dtype_factorize(self, any_real_numpy_dtype):
    dtype = any_real_numpy_dtype
    data = np.array([1, 2, 2, 1], dtype=dtype)
    expected_codes = np.array([0, 1, 1, 0], dtype=np.intp)
    expected_uniques = np.array([1, 2], dtype=dtype)
    codes, uniques = algos.factorize(data)
    tm.assert_numpy_array_equal(codes, expected_codes)
    tm.assert_numpy_array_equal(uniques, expected_uniques)