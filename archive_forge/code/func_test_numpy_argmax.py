from datetime import (
from decimal import Decimal
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core import nanops
from pandas.core.arrays.string_arrow import ArrowStringArrayNumpySemantics
def test_numpy_argmax(self):
    data = np.arange(1, 11)
    ser = Series(data, index=data)
    result = np.argmax(ser)
    expected = np.argmax(data)
    assert result == expected
    result = ser.argmax()
    assert result == expected
    msg = "the 'out' parameter is not supported"
    with pytest.raises(ValueError, match=msg):
        np.argmax(ser, out=data)