from io import StringIO
import numpy as np
from numpy.testing import assert_, assert_allclose, assert_equal
import pandas as pd
import patsy
import pytest
from statsmodels import datasets
from statsmodels.base._constraints import fit_constrained
from statsmodels.discrete.discrete_model import Poisson, Logit
from statsmodels.genmod import families
from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.tools.tools import add_constant
from .results import (
class TestPoissonConstrained1a(CheckPoissonConstrainedMixin):

    @classmethod
    def setup_class(cls):
        cls.res2 = results.results_noexposure_constraint
        cls.idx = [7, 3, 4, 5, 6, 0, 1]
        formula = 'deaths ~ logpyears + smokes + C(agecat)'
        mod = Poisson.from_formula(formula, data=data)
        k_vars = len(mod.exog_names)
        start_params = np.zeros(k_vars)
        start_params[0] = np.log(mod.endog.mean())
        constr = 'C(agecat)[T.4] = C(agecat)[T.5]'
        lc = patsy.DesignInfo(mod.exog_names).linear_constraint(constr)
        cls.res1 = fit_constrained(mod, lc.coefs, lc.constants, start_params=start_params, fit_kwds={'method': 'bfgs', 'disp': 0})
        cls.res1m = mod.fit_constrained(constr, start_params=start_params, method='bfgs', disp=0)

    @pytest.mark.smoke
    def test_summary(self):
        summ = self.res1m.summary()
        assert_('linear equality constraints' in summ.extra_txt)

    @pytest.mark.smoke
    def test_summary2(self):
        summ = self.res1m.summary2()
        assert_('linear equality constraints' in summ.extra_txt[0])