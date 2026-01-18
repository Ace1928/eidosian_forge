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
@pytest.mark.parametrize('level', [0, 'baz'])
def test_unstack_swaplevel_sortlevel(self, level):
    mi = MultiIndex.from_product([[0], ['d', 'c']], names=['bar', 'baz'])
    df = DataFrame([[0, 2], [1, 3]], index=mi, columns=['B', 'A'])
    df.columns.name = 'foo'
    expected = DataFrame([[3, 1, 2, 0]], columns=MultiIndex.from_tuples([('c', 'A'), ('c', 'B'), ('d', 'A'), ('d', 'B')], names=['baz', 'foo']))
    expected.index.name = 'bar'
    result = df.unstack().swaplevel(axis=1).sort_index(axis=1, level=level)
    tm.assert_frame_equal(result, expected)