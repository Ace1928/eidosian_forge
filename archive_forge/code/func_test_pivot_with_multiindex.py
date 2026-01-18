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
@pytest.mark.xfail(reason='MultiIndexed unstack with tuple names fails with KeyError GH#19966')
@pytest.mark.parametrize('method', [True, False])
def test_pivot_with_multiindex(self, method):
    index = Index(data=[0, 1, 2, 3, 4, 5])
    data = [['one', 'A', 1, 'x'], ['one', 'B', 2, 'y'], ['one', 'C', 3, 'z'], ['two', 'A', 4, 'q'], ['two', 'B', 5, 'w'], ['two', 'C', 6, 't']]
    columns = MultiIndex(levels=[['bar', 'baz'], ['first', 'second']], codes=[[0, 0, 1, 1], [0, 1, 0, 1]])
    df = DataFrame(data=data, index=index, columns=columns, dtype='object')
    if method:
        result = df.pivot(index=('bar', 'first'), columns=('bar', 'second'), values=('baz', 'first'))
    else:
        result = pd.pivot(df, index=('bar', 'first'), columns=('bar', 'second'), values=('baz', 'first'))
    data = {'A': Series([1, 4], index=['one', 'two']), 'B': Series([2, 5], index=['one', 'two']), 'C': Series([3, 6], index=['one', 'two'])}
    expected = DataFrame(data)
    tm.assert_frame_equal(result, expected)