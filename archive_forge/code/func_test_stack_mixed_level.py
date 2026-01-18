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
def test_stack_mixed_level(self, future_stack):
    levels = [range(3), [3, 'a', 'b'], [1, 2]]
    df = DataFrame(1, index=levels[0], columns=levels[1])
    result = df.stack(future_stack=future_stack)
    expected = Series(1, index=MultiIndex.from_product(levels[:2]))
    tm.assert_series_equal(result, expected)
    df = DataFrame(1, index=levels[0], columns=MultiIndex.from_product(levels[1:]))
    result = df.stack(1, future_stack=future_stack)
    expected = DataFrame(1, index=MultiIndex.from_product([levels[0], levels[2]]), columns=levels[1])
    tm.assert_frame_equal(result, expected)
    result = df[['a', 'b']].stack(1, future_stack=future_stack)
    expected = expected[['a', 'b']]
    tm.assert_frame_equal(result, expected)