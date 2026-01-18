import numpy as np
from statsmodels.genmod.bayes_mixed_glm import (BinomialBayesMixedGLM,
import pandas as pd
from scipy import sparse
from numpy.testing import assert_allclose, assert_equal
from scipy.optimize import approx_fprime
def test_poisson_formula():
    y, exog_fe, exog_vc, ident = gen_crossed_poisson(10, 10, 1, 0.5)
    for vb in (False, True):
        glmm1 = PoissonBayesMixedGLM(y, exog_fe, exog_vc, ident)
        if vb:
            rslt1 = glmm1.fit_vb()
        else:
            rslt1 = glmm1.fit_map()
        df = pd.DataFrame({'y': y, 'x1': exog_fe[:, 0]})
        z1 = np.zeros(len(y))
        for j, k in enumerate(np.flatnonzero(ident == 0)):
            z1[exog_vc[:, k] == 1] = j
        df['z1'] = z1
        z2 = np.zeros(len(y))
        for j, k in enumerate(np.flatnonzero(ident == 1)):
            z2[exog_vc[:, k] == 1] = j
        df['z2'] = z2
        fml = 'y ~ 0 + x1'
        vc_fml = {}
        vc_fml['z1'] = '0 + C(z1)'
        vc_fml['z2'] = '0 + C(z2)'
        glmm2 = PoissonBayesMixedGLM.from_formula(fml, vc_fml, df)
        if vb:
            rslt2 = glmm2.fit_vb()
        else:
            rslt2 = glmm2.fit_map()
        assert_allclose(rslt1.params, rslt2.params, rtol=1e-05)
        for rslt in (rslt1, rslt2):
            cp = rslt.cov_params()
            p = len(rslt.params)
            if vb:
                assert_equal(cp.shape, np.r_[p,])
                assert_equal(cp > 0, True * np.ones(p))
            else:
                assert_equal(cp.shape, np.r_[p, p])
                np.linalg.cholesky(cp)