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
def test_unstack_multiple_no_empty_columns(self):
    index = MultiIndex.from_tuples([(0, 'foo', 0), (0, 'bar', 0), (1, 'baz', 1), (1, 'qux', 1)])
    s = Series(np.random.default_rng(2).standard_normal(4), index=index)
    unstacked = s.unstack([1, 2])
    expected = unstacked.dropna(axis=1, how='all')
    tm.assert_frame_equal(unstacked, expected)