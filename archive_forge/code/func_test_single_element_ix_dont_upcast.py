from collections import namedtuple
from datetime import (
from decimal import Decimal
import re
import numpy as np
import pytest
from pandas._libs import iNaT
from pandas.errors import (
import pandas.util._test_decorators as td
from pandas.core.dtypes.common import is_integer
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_single_element_ix_dont_upcast(self, float_frame):
    float_frame['E'] = 1
    assert issubclass(float_frame['E'].dtype.type, (int, np.integer))
    result = float_frame.loc[float_frame.index[5], 'E']
    assert is_integer(result)
    df = DataFrame({'a': [1.23]})
    df['b'] = 666
    result = df.loc[0, 'b']
    assert is_integer(result)
    expected = Series([666], [0], name='b')
    result = df.loc[[0], 'b']
    tm.assert_series_equal(result, expected)