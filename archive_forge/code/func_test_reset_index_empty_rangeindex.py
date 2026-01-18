from datetime import datetime
from itertools import product
import numpy as np
import pytest
from pandas.core.dtypes.common import (
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_reset_index_empty_rangeindex(self):
    df = DataFrame(columns=['brand'], dtype=np.int64, index=RangeIndex(0, 0, 1, name='foo'))
    df2 = df.set_index([df.index, 'brand'])
    result = df2.reset_index([1], drop=True)
    tm.assert_frame_equal(result, df[[]], check_index_type=True)