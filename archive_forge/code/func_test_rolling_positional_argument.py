from datetime import (
import numpy as np
import pytest
from pandas.compat import (
from pandas import (
import pandas._testing as tm
from pandas.api.indexers import BaseIndexer
from pandas.core.indexers.objects import VariableOffsetWindowIndexer
from pandas.tseries.offsets import BusinessDay
@pytest.mark.parametrize('grouping,_index', [({'level': 0}, MultiIndex.from_tuples([(0, 0), (0, 0), (1, 1), (1, 1), (1, 1)], names=[None, None])), ({'by': 'X'}, MultiIndex.from_tuples([(0, 0), (1, 0), (2, 1), (3, 1), (4, 1)], names=['X', None]))])
def test_rolling_positional_argument(grouping, _index, raw):

    def scaled_sum(*args):
        if len(args) < 2:
            raise ValueError('The function needs two arguments')
        array, scale = args
        return array.sum() / scale
    df = DataFrame(data={'X': range(5)}, index=[0, 0, 1, 1, 1])
    expected = DataFrame(data={'X': [0.0, 0.5, 1.0, 1.5, 2.0]}, index=_index)
    if 'by' in grouping:
        expected = expected.drop(columns='X', errors='ignore')
    result = df.groupby(**grouping).rolling(1).apply(scaled_sum, raw=raw, args=(2,))
    tm.assert_frame_equal(result, expected)