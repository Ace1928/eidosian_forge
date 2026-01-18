import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_quantile_ea(self, request, obj, index):
    indexer = np.arange(len(index), dtype=np.intp)
    np.random.default_rng(2).shuffle(indexer)
    obj = obj.iloc[indexer]
    qs = [0.5, 0, 1]
    result = self.compute_quantile(obj, qs)
    exp_dtype = index.dtype
    if index.dtype == 'Int64':
        exp_dtype = 'Float64'
    expected = Series([index[4], index[0], index[-1]], dtype=exp_dtype, index=qs, name='A')
    expected = type(obj)(expected)
    tm.assert_equal(result, expected)