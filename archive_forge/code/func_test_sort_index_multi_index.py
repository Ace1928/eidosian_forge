import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_sort_index_multi_index(self):
    df = DataFrame({'a': [3, 1, 2], 'b': [0, 0, 0], 'c': [0, 1, 2], 'd': list('abc')})
    result = df.set_index(list('abc')).sort_index(level=list('ba'))
    expected = DataFrame({'a': [1, 2, 3], 'b': [0, 0, 0], 'c': [1, 2, 0], 'd': list('bca')})
    expected = expected.set_index(list('abc'))
    tm.assert_frame_equal(result, expected)