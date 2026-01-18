import json
import os
import numpy as np
from numpy.testing import (
import pandas as pd
from pandas.testing import assert_frame_equal
import pytest
from statsmodels.datasets import macrodata, sunspots
from statsmodels.regression.linear_model import OLS
import statsmodels.stats.diagnostic as smsdia
import statsmodels.stats.outliers_influence as oi
import statsmodels.stats.sandwich_covariance as sw
from statsmodels.tools.tools import Bunch, add_constant
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.arima.model import ARIMA
@pytest.mark.parametrize('frac', [0.25, 0.5, 0.75])
@pytest.mark.parametrize('order_by', [None, np.arange(500), np.random.choice(500, size=500, replace=False), 'x0', ['x0', 'x2']])
def test_rainbow_smoke_order_by(frac, order_by, reset_randomstate):
    e = pd.DataFrame(np.random.standard_normal((500, 1)))
    x = pd.DataFrame(np.random.standard_normal((500, 3)), columns=[f'x{i}' for i in range(3)])
    y = x @ np.ones((3, 1)) + e
    res = OLS(y, x).fit()
    smsdia.linear_rainbow(res, frac=frac, order_by=order_by)