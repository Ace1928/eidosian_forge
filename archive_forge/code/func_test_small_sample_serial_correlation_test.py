from statsmodels.compat.pandas import MONTH_END
import os
import re
import warnings
import numpy as np
from numpy.testing import (
import pandas as pd
import pytest
from statsmodels.datasets import nile
from statsmodels.tsa.statespace import (
from statsmodels.tsa.statespace.mlemodel import MLEModel, MLEResultsWrapper
from statsmodels.tsa.statespace.tests.results import (
def test_small_sample_serial_correlation_test():
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    niledata = nile.data.load_pandas().data
    niledata.index = pd.date_range('1871-01-01', '1970-01-01', freq='YS')
    mod = SARIMAX(endog=niledata['volume'], order=(1, 0, 1), trend='n', freq=niledata.index.freq)
    res = mod.fit()
    actual = res.test_serial_correlation(method='ljungbox', lags=10, df_adjust=True)[0, :, -1]
    assert_allclose(actual, [14.116, 0.0788], atol=0.001)