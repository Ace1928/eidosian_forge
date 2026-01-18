import numpy as np
from numpy.testing import assert_allclose, assert_equal
from scipy import stats
from scipy.stats import poisson, nbinom
from statsmodels.tools.tools import Bunch
from statsmodels.distributions.discrete import (
def test_ppf_zero(self):
    poisson_ppf = poisson.ppf(5, 1)
    zipoisson_ppf = zipoisson.ppf(5, 1, 0)
    assert_allclose(poisson_ppf, zipoisson_ppf, rtol=1e-12)