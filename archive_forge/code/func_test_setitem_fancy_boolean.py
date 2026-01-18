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
@td.skip_array_manager_invalid_test
def test_setitem_fancy_boolean(self, float_frame):
    frame = float_frame.copy()
    expected = float_frame.copy()
    values = expected.values.copy()
    mask = frame['A'] > 0
    frame.loc[mask] = 0.0
    values[mask.values] = 0.0
    expected = DataFrame(values, index=expected.index, columns=expected.columns)
    tm.assert_frame_equal(frame, expected)
    frame = float_frame.copy()
    expected = float_frame.copy()
    values = expected.values.copy()
    frame.loc[mask, ['A', 'B']] = 0.0
    values[mask.values, :2] = 0.0
    expected = DataFrame(values, index=expected.index, columns=expected.columns)
    tm.assert_frame_equal(frame, expected)