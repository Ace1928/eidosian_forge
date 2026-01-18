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
def test_pivot_periods_with_margins(self):
    df = DataFrame({'a': [1, 1, 2, 2], 'b': [pd.Period('2019Q1'), pd.Period('2019Q2'), pd.Period('2019Q1'), pd.Period('2019Q2')], 'x': 1.0})
    expected = DataFrame(data=1.0, index=Index([1, 2, 'All'], name='a'), columns=Index([pd.Period('2019Q1'), pd.Period('2019Q2'), 'All'], name='b'))
    result = df.pivot_table(index='a', columns='b', values='x', margins=True)
    tm.assert_frame_equal(expected, result)