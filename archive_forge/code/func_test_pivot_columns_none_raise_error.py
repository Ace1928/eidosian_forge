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
def test_pivot_columns_none_raise_error(self):
    df = DataFrame({'col1': ['a', 'b', 'c'], 'col2': [1, 2, 3], 'col3': [1, 2, 3]})
    msg = "pivot\\(\\) missing 1 required keyword-only argument: 'columns'"
    with pytest.raises(TypeError, match=msg):
        df.pivot(index='col1', values='col3')