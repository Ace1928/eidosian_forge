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
def test_sub_alignment_with_duplicate_index(self):
    df1 = DataFrame([1, 2, 3, 4, 5], index=[1, 2, 1, 2, 3])
    df2 = DataFrame([1, 2, 3], index=[1, 2, 3])
    expected = DataFrame([0, 2, 0, 2, 2], index=[1, 1, 2, 2, 3])
    result = df1.sub(df2)
    tm.assert_frame_equal(result, expected)