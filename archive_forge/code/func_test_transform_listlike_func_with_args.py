import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.tests.apply.common import frame_transform_kernels
from pandas.tests.frame.common import zip_frames
def test_transform_listlike_func_with_args():
    df = DataFrame({'x': [1, 2, 3]})

    def foo1(x, a=1, c=0):
        return x + a + c

    def foo2(x, b=2, c=0):
        return x + b + c
    msg = "foo1\\(\\) got an unexpected keyword argument 'b'"
    with pytest.raises(TypeError, match=msg):
        df.transform([foo1, foo2], 0, 3, b=3, c=4)
    result = df.transform([foo1, foo2], 0, 3, c=4)
    expected = DataFrame([[8, 8], [9, 9], [10, 10]], columns=MultiIndex.from_tuples([('x', 'foo1'), ('x', 'foo2')]))
    tm.assert_frame_equal(result, expected)