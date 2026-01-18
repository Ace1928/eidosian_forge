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
def test_sep_return(self):
    data = self.random_data
    n = data.shape[0]
    lagmat, leads = stattools.lagmat(data, 3, trim='none', original='sep')
    expected = np.zeros((n + 3, 4))
    for i in range(4):
        expected[i:i + n, i] = data
    expected_leads = expected[:, :1]
    expected_lags = expected[:, 1:]
    assert_equal(expected_lags, lagmat)
    assert_equal(expected_leads, leads)