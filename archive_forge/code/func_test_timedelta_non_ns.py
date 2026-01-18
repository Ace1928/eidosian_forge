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
def test_timedelta_non_ns(self):
    a = np.array(['2000', '2000', '2001'], dtype='timedelta64[s]')
    result = pd.unique(a)
    expected = np.array([2000, 2001], dtype='timedelta64[s]')
    tm.assert_numpy_array_equal(result, expected)