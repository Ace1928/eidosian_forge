import numpy as np
from numpy.testing import assert_allclose, assert_equal
from scipy import stats
from scipy.stats import poisson, nbinom
from statsmodels.tools.tools import Bunch
from statsmodels.distributions.discrete import (
def test_pmf_zero(self):
    n, p = truncatednegbin.convert_params(5, 0.1, 2)
    nb_pmf = nbinom.pmf(1, n, p) / nbinom.sf(0, n, p)
    tnb_pmf = truncatednegbin.pmf(1, 5, 0.1, 2, 0)
    assert_allclose(nb_pmf, tnb_pmf, rtol=1e-05)