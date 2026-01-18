import numpy as np
from numpy.testing import assert_allclose, assert_equal
from scipy import stats
from scipy.stats import poisson, nbinom
from statsmodels.tools.tools import Bunch
from statsmodels.distributions.discrete import (
def test_ppf_p2(self):
    n, p = zinegbin.convert_params(100, 1, 2)
    nbinom_ppf = nbinom.ppf(0.27, n, p)
    zinbinom_ppf = zinegbin.ppf(0.27, 100, 1, 2, 0)
    assert_allclose(nbinom_ppf, zinbinom_ppf, rtol=1e-12, atol=1e-12)