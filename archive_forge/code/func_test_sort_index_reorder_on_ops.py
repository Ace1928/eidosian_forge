import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_sort_index_reorder_on_ops(self):
    df = DataFrame(np.random.default_rng(2).standard_normal((8, 2)), index=MultiIndex.from_product([['a', 'b'], ['big', 'small'], ['red', 'blu']], names=['letter', 'size', 'color']), columns=['near', 'far'])
    df = df.sort_index()

    def my_func(group):
        group.index = ['newz', 'newa']
        return group
    result = df.groupby(level=['letter', 'size']).apply(my_func).sort_index()
    expected = MultiIndex.from_product([['a', 'b'], ['big', 'small'], ['newa', 'newz']], names=['letter', 'size', None])
    tm.assert_index_equal(result.index, expected)