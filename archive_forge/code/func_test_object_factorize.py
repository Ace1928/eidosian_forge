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
def test_object_factorize(self, writable):
    data = np.array(['a', 'c', None, np.nan, 'a', 'b', NaT, 'c'], dtype=object)
    data.setflags(write=writable)
    expected_codes = np.array([0, 1, -1, -1, 0, 2, -1, 1], dtype=np.intp)
    expected_uniques = np.array(['a', 'c', 'b'], dtype=object)
    codes, uniques = algos.factorize(data)
    tm.assert_numpy_array_equal(codes, expected_codes)
    tm.assert_numpy_array_equal(uniques, expected_uniques)