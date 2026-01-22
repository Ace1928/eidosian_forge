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
def _get_constrained(self, keep_index, keep_index_p):
    mod2 = self.results.model
    mod_cls = mod2.__class__
    init_kwds = mod2._get_init_kwds()
    mod = mod_cls(mod2.endog, mod2.exog[:, keep_index], **init_kwds)
    if self.use_start_params:
        res = mod.fit(start_params=self.results.params[keep_index_p], maxiter=500)
    else:
        res = mod.fit(maxiter=500)
    return res