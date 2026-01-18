from datetime import (
from decimal import Decimal
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core import nanops
from pandas.core.arrays.string_arrow import ArrowStringArrayNumpySemantics
def test_numpy_argmin(self):
    data = np.arange(1, 11)
    s = Series(data, index=data)
    result = np.argmin(s)
    expected = np.argmin(data)
    assert result == expected
    result = s.argmin()
    assert result == expected
    msg = "the 'out' parameter is not supported"
    with pytest.raises(ValueError, match=msg):
        np.argmin(s, out=data)