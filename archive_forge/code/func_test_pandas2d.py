from statsmodels.compat.pandas import (
from datetime import datetime
import numpy as np
from numpy import array, column_stack
from numpy.testing import (
from pandas import DataFrame, concat, date_range
from statsmodels.datasets import macrodata
from statsmodels.tsa.filters._utils import pandas_wrapper
from statsmodels.tsa.filters.bk_filter import bkfilter
from statsmodels.tsa.filters.cf_filter import cffilter
from statsmodels.tsa.filters.filtertools import (
from statsmodels.tsa.filters.hp_filter import hpfilter
def test_pandas2d(self):
    start = datetime(1951, 3, 31)
    end = datetime(1958, 12, 31)
    x = concat((self.data[0], self.data[0]), axis=1)
    res = convolution_filter(x, [[0.75, 0.75], [0.25, 0.25]])
    assert_(res.index[0] == start)
    assert_(res.index[-1] == end)