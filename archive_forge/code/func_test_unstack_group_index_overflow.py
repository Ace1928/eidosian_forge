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
def test_unstack_group_index_overflow(self, future_stack):
    codes = np.tile(np.arange(500), 2)
    level = np.arange(500)
    index = MultiIndex(levels=[level] * 8 + [[0, 1]], codes=[codes] * 8 + [np.arange(2).repeat(500)])
    s = Series(np.arange(1000), index=index)
    result = s.unstack()
    assert result.shape == (500, 2)
    stacked = result.stack(future_stack=future_stack)
    tm.assert_series_equal(s, stacked.reindex(s.index))
    index = MultiIndex(levels=[[0, 1]] + [level] * 8, codes=[np.arange(2).repeat(500)] + [codes] * 8)
    s = Series(np.arange(1000), index=index)
    result = s.unstack(0)
    assert result.shape == (500, 2)
    index = MultiIndex(levels=[level] * 4 + [[0, 1]] + [level] * 4, codes=[codes] * 4 + [np.arange(2).repeat(500)] + [codes] * 4)
    s = Series(np.arange(1000), index=index)
    result = s.unstack(4)
    assert result.shape == (500, 2)