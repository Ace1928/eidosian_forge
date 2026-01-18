from collections import namedtuple
from datetime import (
from decimal import Decimal
import re
import numpy as np
import pytest
from pandas._libs import iNaT
from pandas.errors import (
import pandas.util._test_decorators as td
from pandas.core.dtypes.common import is_integer
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_setitem_None(self, float_frame, using_infer_string):
    float_frame[None] = float_frame['A']
    key = None if not using_infer_string else np.nan
    tm.assert_series_equal(float_frame.iloc[:, -1], float_frame['A'], check_names=False)
    tm.assert_series_equal(float_frame.loc[:, key], float_frame['A'], check_names=False)
    tm.assert_series_equal(float_frame[key], float_frame['A'], check_names=False)