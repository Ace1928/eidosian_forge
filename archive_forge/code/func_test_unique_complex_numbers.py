from datetime import datetime
import struct
import numpy as np
import pytest
from pandas._libs import (
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import CategoricalDtype
import pandas as pd
from pandas import (
import pandas._testing as tm
import pandas.core.algorithms as algos
from pandas.core.arrays import (
import pandas.core.common as com
@pytest.mark.parametrize('array,expected', [([1 + 1j, 0, 1, 1j, 1 + 2j, 1 + 2j], np.array([1 + 1j, 0j, 1 + 0j, 1j, 1 + 2j], dtype=object))])
def test_unique_complex_numbers(self, array, expected):
    msg = 'unique with argument that is not not a Series'
    with tm.assert_produces_warning(FutureWarning, match=msg):
        result = pd.unique(array)
    tm.assert_numpy_array_equal(result, expected)