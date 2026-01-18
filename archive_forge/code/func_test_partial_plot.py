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
@pytest.mark.matplotlib
def test_partial_plot():
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(cur_dir, 'results', 'prediction_from_mgcv.csv')
    data_from_r = pd.read_csv(file_path)
    x = data_from_r.x.values
    y = data_from_r.y.values
    se_from_mgcv = data_from_r.y_est_se
    df = [10]
    degree = [6]
    bsplines = BSplines(x, degree=degree, df=df)
    alpha = 0.03
    glm_gam = GLMGam(y, smoother=bsplines, alpha=alpha)
    res_glm_gam = glm_gam.fit(maxiter=10000, method='bfgs')
    fig = res_glm_gam.plot_partial(0)
    xp, yp = fig.axes[0].get_children()[0].get_data()
    sort_idx = np.argsort(x)
    hat_y, se = res_glm_gam.partial_values(0)
    assert_allclose(xp, x[sort_idx])
    assert_allclose(yp, hat_y[sort_idx])