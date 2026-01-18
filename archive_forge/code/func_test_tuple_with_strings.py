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
@pytest.mark.parametrize('arg ,expected', [(('1', '1', '2'), np.array(['1', '2'], dtype=object)), (('foo',), np.array(['foo'], dtype=object))])
def test_tuple_with_strings(self, arg, expected):
    msg = 'unique with argument that is not not a Series'
    with tm.assert_produces_warning(FutureWarning, match=msg):
        result = pd.unique(arg)
    tm.assert_numpy_array_equal(result, expected)