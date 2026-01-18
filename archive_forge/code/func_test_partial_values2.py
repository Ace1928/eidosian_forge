import os
import numpy as np
from numpy.testing import assert_allclose
import pandas as pd
from scipy.linalg import block_diag
import pytest
from statsmodels.tools.linalg import matrix_sqrt
from statsmodels.gam.smooth_basis import (
from statsmodels.gam.generalized_additive_model import (
from statsmodels.gam.gam_cross_validation.gam_cross_validation import (
from statsmodels.gam.gam_penalties import (UnivariateGamPenalty,
from statsmodels.gam.gam_cross_validation.cross_validators import KFold
from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.genmod.families.family import Gaussian
from statsmodels.genmod.generalized_linear_model import lm
def test_partial_values2():
    np.random.seed(0)
    n = 1000
    x = np.random.uniform(0, 1, (n, 2))
    x = x - x.mean()
    y = x[:, 0] * x[:, 0] + np.random.normal(0, 0.01, n)
    y -= y.mean()
    alpha = 0.0
    bsplines = BSplines(x, degree=[3] * 2, df=[10] * 2, include_intercept=[True, False])
    glm_gam = GLMGam(y, smoother=bsplines, alpha=alpha)
    res_glm_gam = glm_gam.fit(method='pirls', max_start_irls=0, disp=0, maxiter=5000)
    glm = GLM(y, bsplines.basis)
    ex = np.column_stack((bsplines.smoothers[0].basis, np.zeros_like(bsplines.smoothers[1].basis)))
    y_est = res_glm_gam.predict(ex, transform=False)
    y_partial_est, se = res_glm_gam.partial_values(0)
    assert_allclose(y_est, y_partial_est, atol=0.05)
    assert se.min() < 100