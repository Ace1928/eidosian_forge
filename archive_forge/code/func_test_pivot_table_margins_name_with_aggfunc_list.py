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
def test_pivot_table_margins_name_with_aggfunc_list(self):
    margins_name = 'Weekly'
    costs = DataFrame({'item': ['bacon', 'cheese', 'bacon', 'cheese'], 'cost': [2.5, 4.5, 3.2, 3.3], 'day': ['ME', 'ME', 'T', 'T']})
    table = costs.pivot_table(index='item', columns='day', margins=True, margins_name=margins_name, aggfunc=['mean', 'max'])
    ix = Index(['bacon', 'cheese', margins_name], name='item')
    tups = [('mean', 'cost', 'ME'), ('mean', 'cost', 'T'), ('mean', 'cost', margins_name), ('max', 'cost', 'ME'), ('max', 'cost', 'T'), ('max', 'cost', margins_name)]
    cols = MultiIndex.from_tuples(tups, names=[None, None, 'day'])
    expected = DataFrame(table.values, index=ix, columns=cols)
    tm.assert_frame_equal(table, expected)