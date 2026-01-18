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
@pytest.mark.filterwarnings('ignore:Downcasting object dtype arrays:FutureWarning')
@pytest.mark.parametrize('index, columns', [([0, 0, 1, 1], MultiIndex.from_product([[1, 2], ['a', 'b']])), ([0, 0, 2, 3], MultiIndex.from_product([[1, 2], ['a', 'b']])), ([0, 1, 2, 3], MultiIndex.from_product([[1, 2], ['a', 'b']]))])
def test_stack_multi_columns_non_unique_index(self, index, columns, future_stack):
    df = DataFrame(index=index, columns=columns).fillna(1)
    stacked = df.stack(future_stack=future_stack)
    new_index = MultiIndex.from_tuples(stacked.index.to_numpy())
    expected = DataFrame(stacked.to_numpy(), index=new_index, columns=stacked.columns)
    tm.assert_frame_equal(stacked, expected)
    stacked_codes = np.asarray(stacked.index.codes)
    expected_codes = np.asarray(new_index.codes)
    tm.assert_numpy_array_equal(stacked_codes, expected_codes)