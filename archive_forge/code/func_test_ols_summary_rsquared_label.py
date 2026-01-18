import warnings
import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_equal
from statsmodels.iolib.summary2 import summary_col
from statsmodels.tools.tools import add_constant
from statsmodels.regression.linear_model import OLS
def test_ols_summary_rsquared_label():
    x = [1, 5, 7, 3, 5, 2, 5, 3]
    y = [6, 4, 2, 7, 4, 9, 10, 2]
    reg_with_constant = OLS(y, add_constant(x)).fit()
    r2_str = 'R-squared:'
    with pytest.warns(UserWarning):
        assert r2_str in str(reg_with_constant.summary2())
    with pytest.warns(UserWarning):
        assert r2_str in str(reg_with_constant.summary())
    reg_without_constant = OLS(y, x, hasconst=False).fit()
    r2_str = 'R-squared (uncentered):'
    with pytest.warns(UserWarning):
        assert r2_str in str(reg_without_constant.summary2())
    with pytest.warns(UserWarning):
        assert r2_str in str(reg_without_constant.summary())