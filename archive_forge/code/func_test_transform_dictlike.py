import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.tests.apply.common import frame_transform_kernels
from pandas.tests.frame.common import zip_frames
@pytest.mark.parametrize('box', [dict, Series])
def test_transform_dictlike(axis, float_frame, box):
    if axis in (0, 'index'):
        e = float_frame.columns[0]
        expected = float_frame[[e]].transform(np.abs)
    else:
        e = float_frame.index[0]
        expected = float_frame.iloc[[0]].transform(np.abs)
    result = float_frame.transform(box({e: np.abs}), axis=axis)
    tm.assert_frame_equal(result, expected)