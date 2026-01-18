import numpy as np
from numpy.testing import assert_allclose, assert_equal  #noqa
from statsmodels.stats import weightstats
import statsmodels.stats.multivariate as smmv  # pytest cannot import test_xxx
from statsmodels.stats.multivariate import confint_mvmean_fromstats
from statsmodels.tools.testing import Holder
def test_spherical(self):
    cov, nobs = (self.cov, self.nobs)
    p_chi2 = 0.0006422366870356
    chi2 = 21.53275509455011
    stat, pv = smmv.test_cov_spherical(cov, nobs)
    assert_allclose(stat, chi2, rtol=1e-07)
    assert_allclose(pv, p_chi2, rtol=1e-06)