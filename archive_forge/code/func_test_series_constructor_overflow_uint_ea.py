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
@pytest.mark.parametrize('val', [1, 1.0])
def test_series_constructor_overflow_uint_ea(self, val):
    max_val = np.iinfo(np.uint64).max - 1
    result = Series([max_val, val], dtype='UInt64')
    expected = Series(np.array([max_val, 1], dtype='uint64'), dtype='UInt64')
    tm.assert_series_equal(result, expected)