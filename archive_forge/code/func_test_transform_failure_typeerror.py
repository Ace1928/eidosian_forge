import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.tests.apply.common import frame_transform_kernels
from pandas.tests.frame.common import zip_frames
@pytest.mark.parametrize('op', frame_kernels_raise)
def test_transform_failure_typeerror(request, op):
    if op == 'ngroup':
        request.applymarker(pytest.mark.xfail(raises=ValueError, reason='ngroup not valid for NDFrame'))
    df = DataFrame({'A': 3 * [object], 'B': [1, 2, 3]})
    error = TypeError
    msg = '|'.join(["not supported between instances of 'type' and 'type'", 'unsupported operand type'])
    with pytest.raises(error, match=msg):
        df.transform([op])
    with pytest.raises(error, match=msg):
        df.transform({'A': op, 'B': op})
    with pytest.raises(error, match=msg):
        df.transform({'A': [op], 'B': [op]})
    with pytest.raises(error, match=msg):
        df.transform({'A': [op, 'shift'], 'B': [op]})