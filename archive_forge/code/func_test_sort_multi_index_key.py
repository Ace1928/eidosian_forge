import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_sort_multi_index_key(self):
    df = DataFrame({'a': [3, 1, 2], 'b': [0, 0, 0], 'c': [0, 1, 2], 'd': list('abc')}).set_index(list('abc'))
    result = df.sort_index(level=list('ac'), key=lambda x: x)
    expected = DataFrame({'a': [1, 2, 3], 'b': [0, 0, 0], 'c': [1, 2, 0], 'd': list('bca')}).set_index(list('abc'))
    tm.assert_frame_equal(result, expected)
    result = df.sort_index(level=list('ac'), key=lambda x: -x)
    expected = DataFrame({'a': [3, 2, 1], 'b': [0, 0, 0], 'c': [0, 2, 1], 'd': list('acb')}).set_index(list('abc'))
    tm.assert_frame_equal(result, expected)