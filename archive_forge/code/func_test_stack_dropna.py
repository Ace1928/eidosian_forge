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
def test_stack_dropna(self, future_stack):
    df = DataFrame({'A': ['a1', 'a2'], 'B': ['b1', 'b2'], 'C': [1, 1]})
    df = df.set_index(['A', 'B'])
    dropna = False if not future_stack else lib.no_default
    stacked = df.unstack().stack(dropna=dropna, future_stack=future_stack)
    assert len(stacked) > len(stacked.dropna())
    if future_stack:
        with pytest.raises(ValueError, match='dropna must be unspecified'):
            df.unstack().stack(dropna=True, future_stack=future_stack)
    else:
        stacked = df.unstack().stack(dropna=True, future_stack=future_stack)
        tm.assert_frame_equal(stacked, stacked.dropna())