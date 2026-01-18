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
def test_on_index_object(self):
    mindex = MultiIndex.from_arrays([np.arange(5).repeat(5), np.tile(np.arange(5), 5)])
    expected = mindex.values
    expected.sort()
    mindex = mindex.repeat(2)
    result = pd.unique(mindex)
    result.sort()
    tm.assert_almost_equal(result, expected)