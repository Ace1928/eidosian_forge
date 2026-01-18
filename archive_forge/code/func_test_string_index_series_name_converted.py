from datetime import datetime
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_string_index_series_name_converted(self):
    df = DataFrame(np.random.default_rng(2).standard_normal((10, 4)), index=date_range('1/1/2000', periods=10))
    result = df.loc['1/3/2000']
    assert result.name == df.index[2]
    result = df.T['1/3/2000']
    assert result.name == df.index[2]