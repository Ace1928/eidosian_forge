from datetime import datetime
import itertools
import re
import numpy as np
import pytest
from pandas._libs import lib
from pandas.errors import PerformanceWarning
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.reshape import reshape as reshape_lib
@pytest.mark.filterwarnings('ignore:The previous implementation of stack is deprecated')
@pytest.mark.parametrize('levels', itertools.chain.from_iterable((itertools.product(itertools.permutations([0, 1, 2], width), repeat=2) for width in [2, 3])))
@pytest.mark.parametrize('stack_lev', range(2))
@pytest.mark.parametrize('sort', [True, False])
def test_stack_order_with_unsorted_levels(self, levels, stack_lev, sort, future_stack):
    columns = MultiIndex(levels=levels, codes=[[0, 0, 1, 1], [0, 1, 0, 1]])
    df = DataFrame(columns=columns, data=[range(4)])
    kwargs = {} if future_stack else {'sort': sort}
    df_stacked = df.stack(stack_lev, future_stack=future_stack, **kwargs)
    for row in df.index:
        for col in df.columns:
            expected = df.loc[row, col]
            result_row = (row, col[stack_lev])
            result_col = col[1 - stack_lev]
            result = df_stacked.loc[result_row, result_col]
            assert result == expected