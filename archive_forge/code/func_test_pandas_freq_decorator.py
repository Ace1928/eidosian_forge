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
def test_pandas_freq_decorator():
    x = make_dataframe()
    func = pandas_wrapper(dummy_func)
    np.testing.assert_equal(func(x.values), x)
    func = pandas_wrapper(dummy_func_array)
    assert_frame_equal(func(x), x)
    expected = x.rename(columns=dict(zip('ABCD', 'EFGH')))
    func = pandas_wrapper(dummy_func_array, names=list('EFGH'))
    assert_frame_equal(func(x), expected)