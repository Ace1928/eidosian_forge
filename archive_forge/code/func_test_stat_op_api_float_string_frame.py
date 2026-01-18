from datetime import timedelta
from decimal import Decimal
import re
from dateutil.tz import tzlocal
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
from pandas.compat import (
from pandas.compat.numpy import np_version_gt2
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core import (
@pytest.mark.parametrize('axis', [0, 1])
@pytest.mark.parametrize('opname', ['count', 'sum', 'mean', 'product', 'median', 'min', 'max', 'nunique', 'var', 'std', 'sem', pytest.param('skew', marks=td.skip_if_no('scipy')), pytest.param('kurt', marks=td.skip_if_no('scipy'))])
def test_stat_op_api_float_string_frame(self, float_string_frame, axis, opname, using_infer_string):
    if (opname in ('sum', 'min', 'max') and axis == 0 or opname in ('count', 'nunique')) and (not (using_infer_string and opname == 'sum')):
        getattr(float_string_frame, opname)(axis=axis)
    else:
        if opname in ['var', 'std', 'sem', 'skew', 'kurt']:
            msg = "could not convert string to float: 'bar'"
        elif opname == 'product':
            if axis == 1:
                msg = "can't multiply sequence by non-int of type 'float'"
            else:
                msg = "can't multiply sequence by non-int of type 'str'"
        elif opname == 'sum':
            msg = "unsupported operand type\\(s\\) for \\+: 'float' and 'str'"
        elif opname == 'mean':
            if axis == 0:
                msg = '|'.join(["Could not convert \\['.*'\\] to numeric", "Could not convert string '(bar){30}' to numeric"])
            else:
                msg = "unsupported operand type\\(s\\) for \\+: 'float' and 'str'"
        elif opname in ['min', 'max']:
            msg = "'[><]=' not supported between instances of 'float' and 'str'"
        elif opname == 'median':
            msg = re.compile('Cannot convert \\[.*\\] to numeric|does not support', flags=re.S)
        if not isinstance(msg, re.Pattern):
            msg = msg + '|does not support'
        with pytest.raises(TypeError, match=msg):
            getattr(float_string_frame, opname)(axis=axis)
    if opname != 'nunique':
        getattr(float_string_frame, opname)(axis=axis, numeric_only=True)