from datetime import (
from itertools import product
import re
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
from pandas.errors import PerformanceWarning
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.api.types import CategoricalDtype
from pandas.core.reshape import reshape as reshape_lib
from pandas.core.reshape.pivot import pivot_table
def test_pivot_not_changing_index_name(self):
    df = DataFrame({'one': ['a'], 'two': 0, 'three': 1})
    expected = df.copy(deep=True)
    df.pivot(index='one', columns='two', values='three')
    tm.assert_frame_equal(df, expected)