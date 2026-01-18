from datetime import (
import numpy as np
import pytest
from pandas.errors import (
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import NumpyExtensionArray
from pandas.tests.arithmetic.common import (
def test_td64arr_mul_too_short_raises(self, box_with_array):
    idx = TimedeltaIndex(np.arange(5, dtype='int64'))
    idx = tm.box_expected(idx, box_with_array)
    msg = '|'.join(['cannot use operands with types dtype', 'Cannot multiply with unequal lengths', 'Unable to coerce to Series'])
    with pytest.raises(TypeError, match=msg):
        idx * idx[:3]
    with pytest.raises(ValueError, match=msg):
        idx * np.array([1, 2])