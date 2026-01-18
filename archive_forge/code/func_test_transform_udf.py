import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.tests.apply.common import frame_transform_kernels
from pandas.tests.frame.common import zip_frames
@pytest.mark.parametrize('use_apply', [True, False])
def test_transform_udf(axis, float_frame, use_apply, frame_or_series):
    obj = unpack_obj(float_frame, frame_or_series, axis)

    def func(x):
        if use_apply == isinstance(x, frame_or_series):
            raise ValueError
        return x + 1
    result = obj.transform(func, axis=axis)
    expected = obj + 1
    tm.assert_equal(result, expected)