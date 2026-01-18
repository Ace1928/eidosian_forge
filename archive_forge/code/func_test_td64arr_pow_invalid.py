from datetime import (
import numpy as np
import pytest
from pandas.errors import (
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import NumpyExtensionArray
from pandas.tests.arithmetic.common import (
def test_td64arr_pow_invalid(self, scalar_td, box_with_array):
    td1 = Series([timedelta(minutes=5, seconds=3)] * 3)
    td1.iloc[2] = np.nan
    td1 = tm.box_expected(td1, box_with_array)
    pattern = 'operate|unsupported|cannot|not supported'
    with pytest.raises(TypeError, match=pattern):
        scalar_td ** td1
    with pytest.raises(TypeError, match=pattern):
        td1 ** scalar_td