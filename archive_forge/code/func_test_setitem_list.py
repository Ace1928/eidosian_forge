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
def test_setitem_list(self, float_frame):
    float_frame['E'] = 'foo'
    data = float_frame[['A', 'B']]
    float_frame[['B', 'A']] = data
    tm.assert_series_equal(float_frame['B'], data['A'], check_names=False)
    tm.assert_series_equal(float_frame['A'], data['B'], check_names=False)
    msg = 'Columns must be same length as key'
    with pytest.raises(ValueError, match=msg):
        data[['A']] = float_frame[['A', 'B']]
    newcolumndata = range(len(data.index) - 1)
    msg = f'Length of values \\({len(newcolumndata)}\\) does not match length of index \\({len(data)}\\)'
    with pytest.raises(ValueError, match=msg):
        data['A'] = newcolumndata