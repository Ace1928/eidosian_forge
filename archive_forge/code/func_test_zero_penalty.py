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
def test_zero_penalty():
    x, y, poly = multivariate_sample_data()
    alphas = [0, 0]
    gam_gs = GLMGam(y, smoother=poly, alpha=alphas)
    gam_gs_res = gam_gs.fit()
    y_est_gam = gam_gs_res.predict()
    glm = GLM(y, poly.basis).fit()
    y_est = glm.predict()
    assert_allclose(y_est, y_est_gam)