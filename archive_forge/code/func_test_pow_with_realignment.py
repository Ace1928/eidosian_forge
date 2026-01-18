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
def test_pow_with_realignment():
    left = DataFrame({'A': [0, 1, 2]})
    right = DataFrame(index=[0, 1, 2])
    result = left ** right
    expected = DataFrame({'A': [np.nan, 1.0, np.nan]})
    tm.assert_frame_equal(result, expected)