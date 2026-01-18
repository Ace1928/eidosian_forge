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
def test_series_constructor_ea_int_from_bool(self):
    result = Series([True, False, True, pd.NA], dtype='Int64')
    expected = Series([1, 0, 1, pd.NA], dtype='Int64')
    tm.assert_series_equal(result, expected)
    result = Series([True, False, True], dtype='Int64')
    expected = Series([1, 0, 1], dtype='Int64')
    tm.assert_series_equal(result, expected)