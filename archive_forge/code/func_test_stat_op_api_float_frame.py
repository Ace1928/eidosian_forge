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
@pytest.mark.parametrize('opname', ['count', 'sum', 'mean', 'product', 'median', 'min', 'max', 'var', 'std', 'sem', pytest.param('skew', marks=td.skip_if_no('scipy')), pytest.param('kurt', marks=td.skip_if_no('scipy'))])
def test_stat_op_api_float_frame(self, float_frame, axis, opname):
    getattr(float_frame, opname)(axis=axis, numeric_only=False)