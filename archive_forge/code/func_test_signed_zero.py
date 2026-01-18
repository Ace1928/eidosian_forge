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
def test_signed_zero(self):
    a = np.array([-0.0, 0.0])
    result = pd.unique(a)
    expected = np.array([-0.0])
    tm.assert_numpy_array_equal(result, expected)