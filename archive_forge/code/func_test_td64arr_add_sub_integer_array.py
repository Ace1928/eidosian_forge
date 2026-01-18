from datetime import (
import numpy as np
import pytest
from pandas.errors import (
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import NumpyExtensionArray
from pandas.tests.arithmetic.common import (
def test_td64arr_add_sub_integer_array(self, box_with_array):
    box = box_with_array
    xbox = np.ndarray if box is pd.array else box
    rng = timedelta_range('1 days 09:00:00', freq='h', periods=3)
    tdarr = tm.box_expected(rng, box)
    other = tm.box_expected([4, 3, 2], xbox)
    msg = 'Addition/subtraction of integers and integer-arrays'
    assert_invalid_addsub_type(tdarr, other, msg)