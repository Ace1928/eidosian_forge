import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_quantile_ea_scalar(self, request, obj, index):
    indexer = np.arange(len(index), dtype=np.intp)
    np.random.default_rng(2).shuffle(indexer)
    obj = obj.iloc[indexer]
    qs = 0.5
    result = self.compute_quantile(obj, qs)
    exp_dtype = index.dtype
    if index.dtype == 'Int64':
        exp_dtype = 'Float64'
    expected = Series({'A': index[4]}, dtype=exp_dtype, name=0.5)
    if isinstance(obj, Series):
        expected = expected['A']
        assert result == expected
    else:
        tm.assert_series_equal(result, expected)