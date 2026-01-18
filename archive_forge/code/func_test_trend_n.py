from statsmodels.compat.pandas import (
import numpy as np
from numpy.testing import (
import pandas as pd
from pandas.tseries.frequencies import to_offset
import pytest
from statsmodels import regression
from statsmodels.datasets import macrodata
from statsmodels.tsa import stattools
from statsmodels.tsa.tests.results import savedrvs
from statsmodels.tsa.tests.results.datamlw_tls import (
import statsmodels.tsa.tsatools as tools
from statsmodels.tsa.tsatools import vec, vech
def test_trend_n(self):
    assert_equal(tools.add_trend(self.arr_1d, 'n'), self.arr_1d)
    assert tools.add_trend(self.arr_1d, 'n') is not self.arr_1d
    assert_equal(tools.add_trend(self.arr_2d, 'n'), self.arr_2d)
    assert tools.add_trend(self.arr_2d, 'n') is not self.arr_2d