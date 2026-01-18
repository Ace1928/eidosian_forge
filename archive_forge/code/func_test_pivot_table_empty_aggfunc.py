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
@pytest.mark.parametrize('margins', [True, False])
def test_pivot_table_empty_aggfunc(self, margins):
    df = DataFrame({'A': [2, 2, 3, 3, 2], 'id': [5, 6, 7, 8, 9], 'C': ['p', 'q', 'q', 'p', 'q'], 'D': [None, None, None, None, None]})
    result = df.pivot_table(index='A', columns='D', values='id', aggfunc=np.size, margins=margins)
    exp_cols = Index([], name='D')
    expected = DataFrame(index=Index([], dtype='int64', name='A'), columns=exp_cols)
    tm.assert_frame_equal(result, expected)