from collections import deque
from datetime import (
from enum import Enum
import functools
import operator
import re
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.computation import expressions as expr
from pandas.tests.frame.common import (
def test_td64_op_nat_casting(self):
    ser = Series(['NaT', 'NaT'], dtype='timedelta64[ns]')
    df = DataFrame([[1, 2], [3, 4]])
    result = df * ser
    expected = DataFrame({0: ser, 1: ser})
    tm.assert_frame_equal(result, expected)