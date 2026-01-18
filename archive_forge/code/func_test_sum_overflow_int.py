from datetime import (
from decimal import Decimal
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core import nanops
from pandas.core.arrays.string_arrow import ArrowStringArrayNumpySemantics
@pytest.mark.parametrize('use_bottleneck', [True, False])
@pytest.mark.parametrize('dtype', ['int32', 'int64'])
def test_sum_overflow_int(self, use_bottleneck, dtype):
    with pd.option_context('use_bottleneck', use_bottleneck):
        v = np.arange(5000000, dtype=dtype)
        s = Series(v)
        result = s.sum(skipna=False)
        assert int(result) == v.sum(dtype='int64')
        result = s.min(skipna=False)
        assert int(result) == 0
        result = s.max(skipna=False)
        assert int(result) == v[-1]