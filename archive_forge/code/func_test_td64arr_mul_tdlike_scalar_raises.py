from datetime import (
import numpy as np
import pytest
from pandas.errors import (
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import NumpyExtensionArray
from pandas.tests.arithmetic.common import (
def test_td64arr_mul_tdlike_scalar_raises(self, two_hours, box_with_array):
    rng = timedelta_range('1 days', '10 days', name='foo')
    rng = tm.box_expected(rng, box_with_array)
    msg = 'argument must be an integer|cannot use operands with types dtype'
    with pytest.raises(TypeError, match=msg):
        rng * two_hours