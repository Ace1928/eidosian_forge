from datetime import datetime
import numpy as np
import pytest
import pandas.util._test_decorators as td
from pandas.core.dtypes.base import _registry as ea_registry
from pandas.core.dtypes.common import is_object_dtype
from pandas.core.dtypes.dtypes import (
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import SparseArray
from pandas.tseries.offsets import BDay
def test_setitem_list_of_tuples(self, float_frame):
    tuples = list(zip(float_frame['A'], float_frame['B']))
    float_frame['tuples'] = tuples
    result = float_frame['tuples']
    expected = Series(tuples, index=float_frame.index, name='tuples')
    tm.assert_series_equal(result, expected)