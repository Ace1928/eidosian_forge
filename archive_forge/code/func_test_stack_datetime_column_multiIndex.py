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
def test_stack_datetime_column_multiIndex(self, future_stack):
    t = datetime(2014, 1, 1)
    df = DataFrame([1, 2, 3, 4], columns=MultiIndex.from_tuples([(t, 'A', 'B')]))
    warn = None if future_stack else FutureWarning
    msg = 'The previous implementation of stack is deprecated'
    with tm.assert_produces_warning(warn, match=msg):
        result = df.stack(future_stack=future_stack)
    eidx = MultiIndex.from_product([(0, 1, 2, 3), ('B',)])
    ecols = MultiIndex.from_tuples([(t, 'A')])
    expected = DataFrame([1, 2, 3, 4], index=eidx, columns=ecols)
    tm.assert_frame_equal(result, expected)