from __future__ import annotations
from datetime import datetime
import gc
import numpy as np
import pytest
from pandas._libs.tslibs import Timestamp
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import CategoricalDtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import BaseMaskedArray
def test_numpy_repeat(self, simple_index):
    rep = 2
    idx = simple_index
    expected = idx.repeat(rep)
    tm.assert_index_equal(np.repeat(idx, rep), expected)
    msg = "the 'axis' parameter is not supported"
    with pytest.raises(ValueError, match=msg):
        np.repeat(idx, rep, axis=0)