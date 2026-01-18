from datetime import (
import numpy as np
import pytest
import pytz
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tseries.offsets import (
def test_union_with_DatetimeIndex(self, sort):
    i1 = Index(np.arange(0, 20, 2, dtype=np.int64))
    i2 = date_range(start='2012-01-03 00:00:00', periods=10, freq='D')
    i1.union(i2, sort=sort)
    i2.union(i1, sort=sort)