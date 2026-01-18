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
def test_series_from_index_dtype_equal_does_not_copy(self):
    idx = Index([1, 2, 3])
    expected = idx.copy(deep=True)
    ser = Series(idx, dtype='int64')
    ser.iloc[0] = 100
    tm.assert_index_equal(idx, expected)