import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_unstack_tuplename_in_multiindex():
    idx = MultiIndex.from_product([['a', 'b', 'c'], [1, 2, 3]], names=[('A', 'a'), ('B', 'b')])
    ser = Series(1, index=idx)
    result = ser.unstack(('A', 'a'))
    expected = DataFrame([[1, 1, 1], [1, 1, 1], [1, 1, 1]], columns=MultiIndex.from_tuples([('a',), ('b',), ('c',)], names=[('A', 'a')]), index=Index([1, 2, 3], name=('B', 'b')))
    tm.assert_frame_equal(result, expected)