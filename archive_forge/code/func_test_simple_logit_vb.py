import numpy as np
from statsmodels.genmod.bayes_mixed_glm import (BinomialBayesMixedGLM,
import pandas as pd
from scipy import sparse
from numpy.testing import assert_allclose, assert_equal
from scipy.optimize import approx_fprime
def test_simple_logit_vb():
    y, exog_fe, exog_vc, ident = gen_simple_logit(10, 10, 0)
    exog_vc = sparse.csr_matrix(exog_vc)
    glmm1 = BinomialBayesMixedGLM(y, exog_fe, exog_vc, ident, vcp_p=0.5, fe_p=0.5)
    rslt1 = glmm1.fit_map()
    glmm2 = BinomialBayesMixedGLM(y, exog_fe, exog_vc, ident, vcp_p=0.5, fe_p=0.5)
    rslt2 = glmm2.fit_vb(rslt1.params)
    rslt1.summary()
    rslt2.summary()
    assert_allclose(rslt1.params[0:5], np.r_[0.75330405, -0.71643228, -2.49091288, -0.00959806, 0.00450254], rtol=0.0001, atol=0.0001)
    assert_allclose(rslt2.params[0:5], np.r_[0.79338836, -0.7599833, -0.64149356, -0.24772884, 0.10775366], rtol=0.0001, atol=0.0001)
    for rslt in (rslt1, rslt2):
        cp = rslt.cov_params()
        p = len(rslt.params)
        if rslt is rslt1:
            assert_equal(cp.shape, np.r_[p, p])
            np.linalg.cholesky(cp)
        else:
            assert_equal(cp.shape, np.r_[p,])
            assert_equal(cp > 0, True * np.ones(p))