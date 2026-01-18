import numpy as np
import pytest
from pandas.errors import (
from pandas import (
import pandas._testing as tm
def test_sum_object_str_raises(step):
    df = DataFrame({'A': range(5), 'B': range(5, 10), 'C': 'foo'})
    r = df.rolling(window=3, step=step)
    with pytest.raises(DataError, match='Cannot aggregate non-numeric type: object|string'):
        r.sum()