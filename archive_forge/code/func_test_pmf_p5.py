import numpy as np
from numpy.testing import assert_allclose, assert_equal
from scipy import stats
from scipy.stats import poisson, nbinom
from statsmodels.tools.tools import Bunch
from statsmodels.distributions.discrete import (
def test_pmf_p5(self):
    poisson_pmf = poisson.pmf(10, 2)
    genpoisson_pmf_5 = genpoisson_p.pmf(10, 2, 1e-25, 5)
    assert_allclose(poisson_pmf, genpoisson_pmf_5, rtol=1e-12)