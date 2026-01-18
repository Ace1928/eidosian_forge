from itertools import chain
import operator
import numpy as np
import pytest
from pandas.core.dtypes.common import is_number
from pandas import (
import pandas._testing as tm
from pandas.tests.apply.common import (
@pytest.mark.parametrize('op', frame_transform_kernels)
def test_transform_groupby_kernel_frame(request, axis, float_frame, op):
    if op == 'ngroup':
        request.applymarker(pytest.mark.xfail(raises=ValueError, reason='ngroup not valid for NDFrame'))
    args = [0.0] if op == 'fillna' else []
    if axis in (0, 'index'):
        ones = np.ones(float_frame.shape[0])
        msg = "The 'axis' keyword in DataFrame.groupby is deprecated"
    else:
        ones = np.ones(float_frame.shape[1])
        msg = 'DataFrame.groupby with axis=1 is deprecated'
    with tm.assert_produces_warning(FutureWarning, match=msg):
        gb = float_frame.groupby(ones, axis=axis)
    warn = FutureWarning if op == 'fillna' else None
    op_msg = 'DataFrameGroupBy.fillna is deprecated'
    with tm.assert_produces_warning(warn, match=op_msg):
        expected = gb.transform(op, *args)
    result = float_frame.transform(op, axis, *args)
    tm.assert_frame_equal(result, expected)
    assert 'E' not in float_frame.columns
    float_frame['E'] = float_frame['A'].copy()
    assert len(float_frame._mgr.arrays) > 1
    if axis in (0, 'index'):
        ones = np.ones(float_frame.shape[0])
    else:
        ones = np.ones(float_frame.shape[1])
    with tm.assert_produces_warning(FutureWarning, match=msg):
        gb2 = float_frame.groupby(ones, axis=axis)
    warn = FutureWarning if op == 'fillna' else None
    op_msg = 'DataFrameGroupBy.fillna is deprecated'
    with tm.assert_produces_warning(warn, match=op_msg):
        expected2 = gb2.transform(op, *args)
    result2 = float_frame.transform(op, axis, *args)
    tm.assert_frame_equal(result2, expected2)