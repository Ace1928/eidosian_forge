from datetime import (
from decimal import Decimal
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core import nanops
from pandas.core.arrays.string_arrow import ArrowStringArrayNumpySemantics
def test_numpy_minmax_timedelta64(self):
    td = timedelta_range('16815 days', '16820 days', freq='D')
    assert np.min(td) == Timedelta('16815 days')
    assert np.max(td) == Timedelta('16820 days')
    errmsg = "the 'out' parameter is not supported"
    with pytest.raises(ValueError, match=errmsg):
        np.min(td, out=0)
    with pytest.raises(ValueError, match=errmsg):
        np.max(td, out=0)
    assert np.argmin(td) == 0
    assert np.argmax(td) == 5
    errmsg = "the 'out' parameter is not supported"
    with pytest.raises(ValueError, match=errmsg):
        np.argmin(td, out=0)
    with pytest.raises(ValueError, match=errmsg):
        np.argmax(td, out=0)