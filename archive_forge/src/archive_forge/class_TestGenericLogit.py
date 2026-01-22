from statsmodels.compat.pytest import pytest_warns
from statsmodels.compat.pandas import assert_index_equal, assert_series_equal
from statsmodels.compat.platform import (
from statsmodels.compat.scipy import SCIPY_GT_14
import numpy as np
from numpy.testing import (
import pandas as pd
import pytest
import statsmodels.api as sm
from statsmodels.formula.api import glm, ols
import statsmodels.tools._testing as smt
from statsmodels.tools.sm_exceptions import HessianInversionWarning
class TestGenericLogit(CheckGenericMixin):

    def setup_method(self):
        x = self.exog
        nobs = x.shape[0]
        np.random.seed(987689)
        y_bin = (np.random.rand(nobs) < 1.0 / (1 + np.exp(x.sum(1) - x.mean()))).astype(int)
        model = sm.Logit(y_bin, x)
        start_params = np.array([-0.73403806, -1.00901514, -0.97754543, -0.95648212])
        with pytest.warns(FutureWarning, match='Keyword arguments have been passed'):
            self.results = model.fit(start_params=start_params, method='bfgs', disp=0, tol=1e-05)