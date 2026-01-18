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
def test_pivot_table_retains_tz(self):
    dti = date_range('2016-01-01', periods=3, tz='Europe/Amsterdam')
    df = DataFrame({'A': np.random.default_rng(2).standard_normal(3), 'B': np.random.default_rng(2).standard_normal(3), 'C': dti})
    result = df.pivot_table(index=['B', 'C'], dropna=False)
    assert result.index.levels[1].equals(dti)