import datetime
from decimal import Decimal
import operator
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core import ops
@pytest.mark.parametrize('data', [[1, 2, 3], [1.1, 2.2, 3.3], [Timestamp('2011-01-01'), Timestamp('2011-01-02'), pd.NaT], ['x', 'y', 1]])
@pytest.mark.parametrize('dtype', [None, object])
def test_objarr_radd_str_invalid(self, dtype, data, box_with_array):
    ser = Series(data, dtype=dtype)
    ser = tm.box_expected(ser, box_with_array)
    msg = '|'.join(['can only concatenate str', 'did not contain a loop with signature matching types', 'unsupported operand type', 'must be str'])
    with pytest.raises(TypeError, match=msg):
        'foo_' + ser