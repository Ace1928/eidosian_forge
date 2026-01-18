import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.tests.apply.common import frame_transform_kernels
from pandas.tests.frame.common import zip_frames
def test_transform_ufunc(axis, float_frame, frame_or_series):
    obj = unpack_obj(float_frame, frame_or_series, axis)
    with np.errstate(all='ignore'):
        f_sqrt = np.sqrt(obj)
    result = obj.transform(np.sqrt, axis=axis)
    expected = f_sqrt
    tm.assert_equal(result, expected)