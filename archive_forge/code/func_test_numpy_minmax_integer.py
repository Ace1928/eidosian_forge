from datetime import (
from decimal import Decimal
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core import nanops
from pandas.core.arrays.string_arrow import ArrowStringArrayNumpySemantics
def test_numpy_minmax_integer(self):
    idx = Index([1, 2, 3])
    expected = idx.values.max()
    result = np.max(idx)
    assert result == expected
    expected = idx.values.min()
    result = np.min(idx)
    assert result == expected
    errmsg = "the 'out' parameter is not supported"
    with pytest.raises(ValueError, match=errmsg):
        np.min(idx, out=0)
    with pytest.raises(ValueError, match=errmsg):
        np.max(idx, out=0)
    expected = idx.values.argmax()
    result = np.argmax(idx)
    assert result == expected
    expected = idx.values.argmin()
    result = np.argmin(idx)
    assert result == expected
    errmsg = "the 'out' parameter is not supported"
    with pytest.raises(ValueError, match=errmsg):
        np.argmin(idx, out=0)
    with pytest.raises(ValueError, match=errmsg):
        np.argmax(idx, out=0)