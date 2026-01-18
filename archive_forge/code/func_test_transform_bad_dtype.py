import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.tests.apply.common import frame_transform_kernels
from pandas.tests.frame.common import zip_frames
@pytest.mark.parametrize('op', [*frame_kernels_raise, lambda x: x + 1])
def test_transform_bad_dtype(op, frame_or_series, request):
    if op == 'ngroup':
        request.applymarker(pytest.mark.xfail(raises=ValueError, reason='ngroup not valid for NDFrame'))
    obj = DataFrame({'A': 3 * [object]})
    obj = tm.get_obj(obj, frame_or_series)
    error = TypeError
    msg = '|'.join(["not supported between instances of 'type' and 'type'", 'unsupported operand type'])
    with pytest.raises(error, match=msg):
        obj.transform(op)
    with pytest.raises(error, match=msg):
        obj.transform([op])
    with pytest.raises(error, match=msg):
        obj.transform({'A': op})
    with pytest.raises(error, match=msg):
        obj.transform({'A': [op]})