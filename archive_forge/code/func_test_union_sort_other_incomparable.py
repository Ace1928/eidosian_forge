import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.api.types import (
def test_union_sort_other_incomparable():
    idx = MultiIndex.from_product([[1, pd.Timestamp('2000')], ['a', 'b']])
    with tm.assert_produces_warning(RuntimeWarning):
        result = idx.union(idx[:1])
    tm.assert_index_equal(result, idx)
    result = idx.union(idx[:1], sort=False)
    tm.assert_index_equal(result, idx)