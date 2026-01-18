import numpy as np
from statsmodels.genmod.bayes_mixed_glm import (BinomialBayesMixedGLM,
import pandas as pd
from scipy import sparse
from numpy.testing import assert_allclose, assert_equal
from scipy.optimize import approx_fprime
def test_simple_poisson_map():
    y, exog_fe, exog_vc, ident = gen_simple_poisson(10, 10, 0.2)
    exog_vc = sparse.csr_matrix(exog_vc)
    glmm1 = PoissonBayesMixedGLM(y, exog_fe, exog_vc, ident, vcp_p=0.5)
    rslt1 = glmm1.fit_map()
    assert_allclose(glmm1.logposterior_grad(rslt1.params), np.zeros_like(rslt1.params), atol=0.001)
    glmm2 = PoissonBayesMixedGLM(y, exog_fe, exog_vc, ident, vcp_p=0.5)
    rslt2 = glmm2.fit_map()
    assert_allclose(rslt1.params, rslt2.params, atol=0.0001)
    for linear in (False, True):
        for exog in (None, exog_fe):
            pr1 = rslt1.predict(linear=linear, exog=exog)
            pr2 = rslt2.predict(linear=linear, exog=exog)
            pr3 = glmm1.predict(rslt1.params, linear=linear, exog=exog)
            pr4 = glmm2.predict(rslt2.params, linear=linear, exog=exog)
            assert_allclose(pr1, pr2, rtol=1e-05)
            assert_allclose(pr2, pr3, rtol=1e-05)
            assert_allclose(pr3, pr4, rtol=1e-05)
            if not linear:
                assert_equal(pr1.min() >= 0, True)
                assert_equal(pr2.min() >= 0, True)
                assert_equal(pr3.min() >= 0, True)
    for rslt in (rslt1, rslt2):
        cp = rslt.cov_params()
        p = len(rslt.params)
        assert_equal(cp.shape, np.r_[p, p])
        np.linalg.cholesky(cp)