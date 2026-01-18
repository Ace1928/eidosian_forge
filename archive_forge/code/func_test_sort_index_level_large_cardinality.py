import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_sort_index_level_large_cardinality(self):
    index = MultiIndex.from_arrays([np.arange(4000)] * 3)
    df = DataFrame(np.random.default_rng(2).standard_normal(4000).astype('int64'), index=index)
    result = df.sort_index(level=0)
    assert result.index._lexsort_depth == 3
    index = MultiIndex.from_arrays([np.arange(4000)] * 3)
    df = DataFrame(np.random.default_rng(2).standard_normal(4000).astype('int32'), index=index)
    result = df.sort_index(level=0)
    assert (result.dtypes.values == df.dtypes.values).all()
    assert result.index._lexsort_depth == 3