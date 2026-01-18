import numpy as np
import pytest
from scipy.stats import bootstrap, monte_carlo_test, permutation_test
from numpy.testing import assert_allclose, assert_equal, suppress_warnings
from scipy import stats
from scipy import special
from .. import _resampling as _resampling
from scipy._lib._util import rng_integers
from scipy.optimize import root
def test_p_never_zero(self):
    rng = np.random.default_rng(2190176673029737545)
    x = np.zeros(100)
    res = monte_carlo_test(x, rng.random, np.mean, vectorized=True, alternative='less')
    assert res.pvalue == 0.0001