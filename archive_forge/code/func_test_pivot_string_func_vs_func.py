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
@pytest.mark.parametrize('f, f_numpy', [('sum', np.sum), ('mean', np.mean), ('std', np.std), (['sum', 'mean'], [np.sum, np.mean]), (['sum', 'std'], [np.sum, np.std]), (['std', 'mean'], [np.std, np.mean])])
def test_pivot_string_func_vs_func(self, f, f_numpy, data):
    data = data.drop(columns='C')
    result = pivot_table(data, index='A', columns='B', aggfunc=f)
    ops = '|'.join(f) if isinstance(f, list) else f
    msg = f'using DataFrameGroupBy.[{ops}]'
    with tm.assert_produces_warning(FutureWarning, match=msg):
        expected = pivot_table(data, index='A', columns='B', aggfunc=f_numpy)
    tm.assert_frame_equal(result, expected)