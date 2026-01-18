from statsmodels.compat.python import lrange
import warnings
import numpy as np
from numpy.testing import (
import pandas as pd
import pytest
from scipy.linalg import toeplitz
from scipy.stats import t as student_t
from statsmodels.datasets import longley
from statsmodels.regression.linear_model import (
from statsmodels.tools.tools import add_constant
def test_wls_missing():
    from statsmodels.datasets.ccard import load
    data = load()
    endog = data.endog
    endog[[10, 25]] = np.nan
    mod = WLS(data.endog, data.exog, weights=1 / data.exog.iloc[:, 2], missing='drop')
    assert_equal(mod.endog.shape[0], 70)
    assert_equal(mod.exog.shape[0], 70)
    assert_equal(mod.weights.shape[0], 70)