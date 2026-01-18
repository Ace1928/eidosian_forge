import numpy as np
import pytest
from pandas._libs import lib
from pandas.core.dtypes.common import ensure_platform_int
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.groupby import get_groupby_method_args
def test_transform_fastpath_raises():
    df = DataFrame({'A': [1, 1, 2, 2], 'B': [1, -1, 1, 2]})
    gb = df.groupby('A')

    def func(grp):
        if grp.ndim == 2:
            raise NotImplementedError("Don't cross the streams")
        return grp * 2
    obj = gb._obj_with_exclusions
    gen = gb._grouper.get_iterator(obj, axis=gb.axis)
    fast_path, slow_path = gb._define_paths(func)
    _, group = next(gen)
    with pytest.raises(NotImplementedError, match="Don't cross the streams"):
        fast_path(group)
    result = gb.transform(func)
    expected = DataFrame([2, -2, 2, 4], columns=['B'])
    tm.assert_frame_equal(result, expected)