from datetime import datetime
import itertools
import re
import numpy as np
import pytest
from pandas._libs import lib
from pandas.errors import PerformanceWarning
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.reshape import reshape as reshape_lib
@pytest.mark.filterwarnings('ignore:The previous implementation of stack is deprecated')
@pytest.mark.parametrize('vals1, vals2, dtype1, dtype2, expected_dtype', [([1, 2], [3.0, 4.0], 'Int64', 'Float64', 'Float64'), ([1, 2], ['foo', 'bar'], 'Int64', 'string', 'object')])
def test_stack_multi_columns_mixed_extension_types(self, vals1, vals2, dtype1, dtype2, expected_dtype, future_stack):
    df = DataFrame({('A', 1): Series(vals1, dtype=dtype1), ('A', 2): Series(vals2, dtype=dtype2)})
    result = df.stack(future_stack=future_stack)
    expected = df.astype(object).stack(future_stack=future_stack).astype(expected_dtype)
    tm.assert_frame_equal(result, expected)