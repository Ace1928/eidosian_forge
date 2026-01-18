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
def test_series_both(self):
    expected = pd.DataFrame(index=self.series.index, columns=['cpi', 'cpi.L.1', 'cpi.L.2', 'cpi.L.3'])
    expected['cpi'] = self.series
    for lag in range(1, 4):
        expected['cpi.L.' + str(int(lag))] = self.series.shift(lag)
    expected = expected.iloc[3:]
    both = stattools.lagmat(self.series, 3, trim='both', original='in', use_pandas=True)
    assert_frame_equal(both, expected)
    lags = stattools.lagmat(self.series, 3, trim='both', original='ex', use_pandas=True)
    assert_frame_equal(lags, expected.iloc[:, 1:])
    lags, lead = stattools.lagmat(self.series, 3, trim='both', original='sep', use_pandas=True)
    assert_frame_equal(lead, expected.iloc[:, :1])
    assert_frame_equal(lags, expected.iloc[:, 1:])