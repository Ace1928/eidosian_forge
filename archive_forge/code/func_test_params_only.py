from io import BytesIO
from itertools import product
import warnings
import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from statsmodels import tools
from statsmodels.regression.linear_model import WLS
from statsmodels.regression.rolling import RollingWLS, RollingOLS
@pytest.mark.parametrize('method', ['inv', 'lstsq', 'pinv'])
def test_params_only(basic_data, method):
    y, x, _ = basic_data
    mod = RollingOLS(y, x, 150)
    res = mod.fit(method=method, params_only=False)
    res_params_only = mod.fit(method=method, params_only=True)
    assert_allclose(res_params_only.params, res.params)