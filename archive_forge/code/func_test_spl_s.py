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
def test_spl_s():
    spl_s_R = [[0, 0, 0.0, 0.0, 0.0, 0.0], [0, 0, 0.0, 0.0, 0.0, 0.0], [0, 0, 0.0014, 0.0002, -0.001133333, -0.001], [0, 0, 0.0002, 0.002733333, 0.001666667, -0.001133333], [0, 0, -0.001133333, 0.001666667, 0.002733333, 0.0002], [0, 0, -0.001, -0.001133333, 0.0002, 0.0014]]
    np.random.seed(1)
    x = np.random.normal(0, 1, 10)
    xk = np.array([0.2, 0.4, 0.6, 0.8])
    cs = UnivariateCubicSplines(x, df=4)
    cs.knots = xk
    spl_s = cs._splines_s()
    assert_allclose(spl_s_R, spl_s, atol=4e-10)