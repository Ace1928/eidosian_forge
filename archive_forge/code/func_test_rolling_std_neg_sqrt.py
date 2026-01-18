from datetime import (
import numpy as np
import pytest
from pandas.compat import (
from pandas import (
import pandas._testing as tm
from pandas.api.indexers import BaseIndexer
from pandas.core.indexers.objects import VariableOffsetWindowIndexer
from pandas.tseries.offsets import BusinessDay
def test_rolling_std_neg_sqrt():
    a = Series([0.0011448196318903589, 0.00028718669878572767, 0.00028718669878572767, 0.00028718669878572767, 0.00028718669878572767])
    b = a.rolling(window=3).std()
    assert np.isfinite(b[2:]).all()
    b = a.ewm(span=3).std()
    assert np.isfinite(b[2:]).all()