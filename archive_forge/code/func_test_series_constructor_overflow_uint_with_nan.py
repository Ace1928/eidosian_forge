from collections import OrderedDict
from collections.abc import Iterator
from datetime import (
from dateutil.tz import tzoffset
import numpy as np
from numpy import ma
import pytest
from pandas._libs import (
from pandas.compat.numpy import np_version_gt2
from pandas.errors import IntCastingNaNError
import pandas.util._test_decorators as td
from pandas.core.dtypes.dtypes import CategoricalDtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import (
from pandas.core.internals.blocks import NumpyBlock
def test_series_constructor_overflow_uint_with_nan(self):
    max_val = np.iinfo(np.uint64).max - 1
    result = Series([max_val, np.nan], dtype='UInt64')
    expected = Series(IntegerArray(np.array([max_val, 1], dtype='uint64'), np.array([0, 1], dtype=np.bool_)))
    tm.assert_series_equal(result, expected)