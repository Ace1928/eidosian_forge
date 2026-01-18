import numpy as np
from numpy.testing import assert_allclose, assert_equal
import pytest
from statsmodels.regression.linear_model import OLS
import statsmodels.stats.power as smpwr
import statsmodels.stats.oneway as smo  # needed for function with `test`
from statsmodels.stats.oneway import (
from statsmodels.stats.robust_compare import scale_transform
from statsmodels.stats.contrast import (
@pytest.mark.parametrize('center', ['median', 'mean', 'trimmed'])
def test_scale_transform(center):
    x = np.random.randn(5, 3)
    xt = scale_transform(x, center=center, transform='abs', trim_frac=0.2, axis=0)
    xtt = scale_transform(x.T, center=center, transform='abs', trim_frac=0.2, axis=1)
    assert_allclose(xt.T, xtt, rtol=1e-13)
    xt0 = scale_transform(x[:, 0], center=center, transform='abs', trim_frac=0.2)
    assert_allclose(xt0, xt[:, 0], rtol=1e-13)
    assert_allclose(xt0, xtt[0, :], rtol=1e-13)