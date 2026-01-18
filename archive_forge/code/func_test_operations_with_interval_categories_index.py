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
def test_operations_with_interval_categories_index(self, all_arithmetic_operators):
    op = all_arithmetic_operators
    ind = pd.CategoricalIndex(pd.interval_range(start=0.0, end=2.0))
    data = [1, 2]
    df = DataFrame([data], columns=ind)
    num = 10
    result = getattr(df, op)(num)
    expected = DataFrame([[getattr(n, op)(num) for n in data]], columns=ind)
    tm.assert_frame_equal(result, expected)