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
def test_sub_object(self):
    index = pd.Index([Decimal(1), Decimal(2)])
    expected = pd.Index([Decimal(0), Decimal(1)])
    result = index - Decimal(1)
    tm.assert_index_equal(result, expected)
    result = index - pd.Index([Decimal(1), Decimal(1)])
    tm.assert_index_equal(result, expected)
    msg = 'unsupported operand type'
    with pytest.raises(TypeError, match=msg):
        index - 'foo'
    with pytest.raises(TypeError, match=msg):
        index - np.array([2, 'foo'], dtype=object)