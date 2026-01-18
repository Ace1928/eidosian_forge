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
def test_uint64_factorize(self, writable):
    data = np.array([2 ** 64 - 1, 1, 2 ** 64 - 1], dtype=np.uint64)
    data.setflags(write=writable)
    expected_codes = np.array([0, 1, 0], dtype=np.intp)
    expected_uniques = np.array([2 ** 64 - 1, 1], dtype=np.uint64)
    codes, uniques = algos.factorize(data)
    tm.assert_numpy_array_equal(codes, expected_codes)
    tm.assert_numpy_array_equal(uniques, expected_uniques)