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
def test_small_skip(reset_randomstate):
    y = np.random.standard_normal(10)
    x = np.random.standard_normal((10, 3))
    x[:3] = x[:1]
    with pytest.raises(ValueError, match='The initial regressor matrix,'):
        smsdia.recursive_olsresiduals(OLS(y, x).fit())