import numpy as np
from numpy.testing import assert_allclose, assert_equal
from scipy import stats
from scipy.stats import poisson, nbinom
from statsmodels.tools.tools import Bunch
from statsmodels.distributions.discrete import (
def test_pmf_p2(self):
    n, p = zinegbin.convert_params(30, 0.1, 2)
    nb_pmf = nbinom.pmf(100, n, p)
    tnb_pmf = zinegbin.pmf(100, 30, 0.1, 2, 0.01)
    assert_allclose(nb_pmf, tnb_pmf, rtol=1e-05, atol=1e-05)