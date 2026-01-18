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
def test_train_test_smoothers():
    n = 6
    x = np.zeros(shape=(n, 2))
    x[:, 0] = range(6)
    x[:, 1] = range(6, 12)
    poly = PolynomialSmoother(x, degrees=[3, 3])
    train_index = list(range(3))
    test_index = list(range(3, 6))
    train_smoother, test_smoother = _split_train_test_smoothers(poly.x, poly, train_index, test_index)
    expected_train_basis = [[0.0, 0.0, 0.0, 6.0, 36.0, 216.0], [1.0, 1.0, 1.0, 7.0, 49.0, 343.0], [2.0, 4.0, 8.0, 8.0, 64.0, 512.0]]
    assert_allclose(train_smoother.basis, expected_train_basis)
    expected_test_basis = [[3.0, 9.0, 27.0, 9.0, 81.0, 729.0], [4.0, 16.0, 64.0, 10.0, 100.0, 1000.0], [5.0, 25.0, 125.0, 11.0, 121.0, 1331.0]]
    assert_allclose(test_smoother.basis, expected_test_basis)