from datetime import datetime
from hypothesis import given
import numpy as np
import pytest
from pandas.core.dtypes.common import is_scalar
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas._testing._hypothesis import OPTIONAL_ONE_OF_ALL
@pytest.mark.filterwarnings('ignore::DeprecationWarning')
def test_where_invalid(self):
    df = DataFrame(np.random.default_rng(2).standard_normal((5, 3)), columns=['A', 'B', 'C'])
    cond = df > 0
    err1 = (df + 1).values[0:2, :]
    msg = 'other must be the same shape as self when an ndarray'
    with pytest.raises(ValueError, match=msg):
        df.where(cond, err1)
    err2 = cond.iloc[:2, :].values
    other1 = _safe_add(df)
    msg = 'Array conditional must be same shape as self'
    with pytest.raises(ValueError, match=msg):
        df.where(err2, other1)
    with pytest.raises(ValueError, match=msg):
        df.mask(True)
    with pytest.raises(ValueError, match=msg):
        df.mask(0)