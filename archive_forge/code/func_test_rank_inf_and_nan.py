from datetime import (
import numpy as np
import pytest
from pandas._libs.algos import (
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('contents,dtype', [([-np.inf, -50, -1, -1e-20, -1e-25, -1e-50, 0, 1e-40, 1e-20, 1e-10, 2, 40, np.inf], 'float64'), ([-np.inf, -50, -1, -1e-20, -1e-25, -1e-45, 0, 1e-40, 1e-20, 1e-10, 2, 40, np.inf], 'float32'), ([np.iinfo(np.uint8).min, 1, 2, 100, np.iinfo(np.uint8).max], 'uint8'), ([np.iinfo(np.int64).min, -100, 0, 1, 9999, 100000, 10000000000.0, np.iinfo(np.int64).max], 'int64'), ([NegInfinity(), '1', 'A', 'BA', 'Ba', 'C', Infinity()], 'object'), ([datetime(2001, 1, 1), datetime(2001, 1, 2), datetime(2001, 1, 5)], 'datetime64')])
def test_rank_inf_and_nan(self, contents, dtype, frame_or_series):
    dtype_na_map = {'float64': np.nan, 'float32': np.nan, 'object': None, 'datetime64': np.datetime64('nat')}
    values = np.array(contents, dtype=dtype)
    exp_order = np.array(range(len(values)), dtype='float64') + 1.0
    if dtype in dtype_na_map:
        na_value = dtype_na_map[dtype]
        nan_indices = np.random.default_rng(2).choice(range(len(values)), 5)
        values = np.insert(values, nan_indices, na_value)
        exp_order = np.insert(exp_order, nan_indices, np.nan)
    random_order = np.random.default_rng(2).permutation(len(values))
    obj = frame_or_series(values[random_order])
    expected = frame_or_series(exp_order[random_order], dtype='float64')
    result = obj.rank()
    tm.assert_equal(result, expected)