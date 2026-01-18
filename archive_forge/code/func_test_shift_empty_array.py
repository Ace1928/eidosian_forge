import inspect
import operator
import numpy as np
import pytest
from pandas._typing import Dtype
from pandas.core.dtypes.common import is_bool_dtype
from pandas.core.dtypes.dtypes import NumpyEADtype
from pandas.core.dtypes.missing import na_value_for_dtype
import pandas as pd
import pandas._testing as tm
from pandas.core.sorting import nargsort
@pytest.mark.parametrize('periods', [-4, -1, 0, 1, 4])
def test_shift_empty_array(self, data, periods):
    empty = data[:0]
    result = empty.shift(periods)
    expected = empty
    tm.assert_extension_array_equal(result, expected)