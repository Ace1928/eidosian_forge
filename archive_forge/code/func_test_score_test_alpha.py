import warnings
import numpy as np
from numpy.testing import assert_allclose, assert_equal
import pytest
from statsmodels.tools.tools import add_constant
from statsmodels.base._prediction_inference import PredictionResultsMonotonic
from statsmodels.discrete.discrete_model import (
from statsmodels.discrete.count_model import (
from statsmodels.sandbox.regression.tests.test_gmm_poisson import DATA
from .results import results_predict as resp
def test_score_test_alpha(self):
    modr = self.klass(endog, exog.values[:, :-1])
    resr = modr.fit(method='newton', maxiter=300)
    params_restr = np.concatenate([resr.params[:], [0]])
    r_matrix = np.zeros((1, len(params_restr)))
    r_matrix[0, -1] = 1
    np.random.seed(987125643)
    exog_extra = 0.01 * np.random.randn(endog.shape[0])
    from statsmodels.base._parameter_inference import score_test, _scorehess_extra
    sh = _scorehess_extra(resr, exog_extra=None, exog2_extra=exog_extra, hess_kwds=None)
    assert not np.isnan(sh[0]).any()
    sc2 = score_test(resr, exog_extra=(None, exog_extra))
    assert sc2[1] > 0.01
    sc2_hc = score_test(resr, exog_extra=(None, exog_extra), cov_type='HC0')
    assert sc2_hc[1] > 0.01