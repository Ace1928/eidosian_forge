from statsmodels.compat.python import lrange
import numpy as np
from numpy.testing import assert_almost_equal, assert_equal
from scipy import stats
import pytest
from statsmodels.sandbox.gam import AdditiveModel
from statsmodels.sandbox.gam import Model as GAM #?
from statsmodels.genmod.families import family, links
from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.regression.linear_model import OLS
class BaseAM:

    @classmethod
    def setup_class(cls):
        order = 3
        nobs = 200
        lb, ub = (-3.5, 3)
        x1 = np.linspace(lb, ub, nobs)
        x2 = np.sin(2 * x1)
        x = np.column_stack((x1 / x1.max() * 1, 1.0 * x2))
        exog = (x[:, :, None] ** np.arange(order + 1)[None, None, :]).reshape(nobs, -1)
        idx = lrange((order + 1) * 2)
        del idx[order + 1]
        exog_reduced = exog[:, idx]
        y_true = exog.sum(1)
        cls.nobs = nobs
        cls.y_true, cls.x, cls.exog = (y_true, x, exog_reduced)