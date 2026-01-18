import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.tests.apply.common import frame_transform_kernels
from pandas.tests.frame.common import zip_frames
def test_transform_dictlike_mixed():
    df = DataFrame({'a': [1, 2], 'b': [1, 4], 'c': [1, 4]})
    result = df.transform({'b': ['sqrt', 'abs'], 'c': 'sqrt'})
    expected = DataFrame([[1.0, 1, 1.0], [2.0, 4, 2.0]], columns=MultiIndex([('b', 'c'), ('sqrt', 'abs')], [(0, 0, 1), (0, 1, 0)]))
    tm.assert_frame_equal(result, expected)