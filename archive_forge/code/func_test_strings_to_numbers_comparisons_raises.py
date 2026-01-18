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
def test_strings_to_numbers_comparisons_raises(self, compare_operators_no_eq_ne):
    df = DataFrame({x: {'x': 'foo', 'y': 'bar', 'z': 'baz'} for x in ['a', 'b', 'c']})
    f = getattr(operator, compare_operators_no_eq_ne)
    msg = "'[<>]=?' not supported between instances of 'str' and 'int'"
    with pytest.raises(TypeError, match=msg):
        f(df, 0)