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
class TestGLMPoissonConstrained1b(CheckPoissonConstrainedMixin):

    @classmethod
    def setup_class(cls):
        from statsmodels.base._constraints import fit_constrained
        from statsmodels.genmod import families
        from statsmodels.genmod.generalized_linear_model import GLM
        cls.res2 = results.results_exposure_constraint
        cls.idx = [6, 2, 3, 4, 5, 0]
        formula = 'deaths ~ smokes + C(agecat)'
        mod = GLM.from_formula(formula, data=data, family=families.Poisson(), offset=np.log(data['pyears'].values))
        constr = 'C(agecat)[T.4] = C(agecat)[T.5]'
        lc = patsy.DesignInfo(mod.exog_names).linear_constraint(constr)
        cls.res1 = fit_constrained(mod, lc.coefs, lc.constants, fit_kwds={'atol': 1e-10})
        cls.constraints = lc
        cls.res1m = mod.fit_constrained(constr, atol=1e-10)._results

    def test_compare_glm_poisson(self):
        res1 = self.res1m
        res2 = self.res2
        formula = 'deaths ~ smokes + C(agecat)'
        mod = Poisson.from_formula(formula, data=data, exposure=data['pyears'].values)
        constr = 'C(agecat)[T.4] = C(agecat)[T.5]'
        res2 = mod.fit_constrained(constr, start_params=self.res1m.params, method='newton', warn_convergence=False, disp=0)
        assert_allclose(res1.params, res2.params, rtol=1e-12)
        assert_allclose(res1.bse, res2.bse, rtol=1e-11)
        predicted = res1.predict()
        assert_allclose(predicted, res2.predict(), rtol=1e-10)
        assert_allclose(res1.mu, predicted, rtol=1e-10)
        assert_allclose(res1.fittedvalues, predicted, rtol=1e-10)
        assert_allclose(res2.predict(which='linear'), res2.predict(which='linear'), rtol=1e-10)