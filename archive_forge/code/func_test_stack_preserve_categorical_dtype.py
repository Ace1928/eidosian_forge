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
@pytest.mark.parametrize('ordered', [False, True])
def test_stack_preserve_categorical_dtype(self, ordered, future_stack):
    cidx = pd.CategoricalIndex(list('yxz'), categories=list('xyz'), ordered=ordered)
    df = DataFrame([[10, 11, 12]], columns=cidx)
    result = df.stack(future_stack=future_stack)
    midx = MultiIndex.from_product([df.index, cidx])
    expected = Series([10, 11, 12], index=midx)
    tm.assert_series_equal(result, expected)