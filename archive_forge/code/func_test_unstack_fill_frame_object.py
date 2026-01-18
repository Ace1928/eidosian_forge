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
def test_unstack_fill_frame_object():
    data = Series(['a', 'b', 'c', 'a'], dtype='object')
    data.index = MultiIndex.from_tuples([('x', 'a'), ('x', 'b'), ('y', 'b'), ('z', 'a')])
    result = data.unstack()
    expected = DataFrame({'a': ['a', np.nan, 'a'], 'b': ['b', 'c', np.nan]}, index=list('xyz'), dtype=object)
    tm.assert_frame_equal(result, expected)
    result = data.unstack(fill_value='d')
    expected = DataFrame({'a': ['a', 'd', 'a'], 'b': ['b', 'c', 'd']}, index=list('xyz'), dtype=object)
    tm.assert_frame_equal(result, expected)