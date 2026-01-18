import warnings
import numpy as np
from numpy.testing import assert_allclose, assert_almost_equal
import pytest
from scipy import stats
import statsmodels.api as sm
from statsmodels.robust import norms
from statsmodels.robust.robust_linear_model import RLM
from statsmodels.robust.scale import HuberScale
def test_perfect_fit(perfect_fit_data, norm):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        res = RLM(perfect_fit_data.endog, perfect_fit_data.exog, M=norm).fit()
    assert_allclose(res.params, np.array([0, 1, 1]), atol=1e-08)