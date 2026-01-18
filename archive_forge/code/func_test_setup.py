import warnings
import numpy as np
from numpy.testing import assert_allclose, assert_equal
import pandas as pd
import pytest
import scipy.stats as stats
from statsmodels.discrete.discrete_model import Logit
from statsmodels.miscmodels.ordinal_model import OrderedModel
from statsmodels.tools.sm_exceptions import HessianInversionWarning
from statsmodels.tools.tools import add_constant
from .results.results_ordinal_model import data_store as ds
def test_setup(self):
    data = self.data
    resp = self.resp
    fittedvalues = resp.predict()
    formulas = ['apply ~ 1 + pared + public + gpa + C(dummy)', 'apply ~ pared + public + gpa + C(dummy)']
    for formula in formulas:
        modf1 = OrderedModel.from_formula(formula, data, distr='logit')
        resf1 = modf1.fit(method='bfgs')
        summf1 = resf1.summary()
        summf1_str = str(summf1)
        assert resf1.model.exog_names == resp.model.exog_names
        assert resf1.model.data.param_names == resp.model.exog_names
        assert all((name in summf1_str for name in resp.model.data.param_names))
        assert_allclose(resf1.predict(data[:5]), fittedvalues[:5])
    formula = 'apply ~ 0 + pared + public + gpa + C(dummy)'
    with pytest.raises(ValueError, match='not be a constant'):
        OrderedModel.from_formula(formula, data, distr='logit')
    modf2 = OrderedModel.from_formula(formula, data, distr='logit', hasconst=False)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', HessianInversionWarning)
        resf2 = modf2.fit(method='bfgs')
    assert_allclose(resf2.predict(data[:5]), fittedvalues[:5], rtol=0.0001)