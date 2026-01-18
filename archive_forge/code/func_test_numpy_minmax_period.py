from datetime import (
from decimal import Decimal
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core import nanops
from pandas.core.arrays.string_arrow import ArrowStringArrayNumpySemantics
def test_numpy_minmax_period(self):
    pr = period_range(start='2016-01-15', end='2016-01-20')
    assert np.min(pr) == Period('2016-01-15', freq='D')
    assert np.max(pr) == Period('2016-01-20', freq='D')
    errmsg = "the 'out' parameter is not supported"
    with pytest.raises(ValueError, match=errmsg):
        np.min(pr, out=0)
    with pytest.raises(ValueError, match=errmsg):
        np.max(pr, out=0)
    assert np.argmin(pr) == 0
    assert np.argmax(pr) == 5
    errmsg = "the 'out' parameter is not supported"
    with pytest.raises(ValueError, match=errmsg):
        np.argmin(pr, out=0)
    with pytest.raises(ValueError, match=errmsg):
        np.argmax(pr, out=0)