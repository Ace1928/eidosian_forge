import pytest
import itertools
from scipy.stats import (betabinom, betanbinom, hypergeom, nhypergeom,
import numpy as np
from numpy.testing import (
from scipy.special import binom as special_binom
from scipy.optimize import root_scalar
from scipy.integrate import quad
def test_zipfian_R(self):
    np.random.seed(0)
    k = np.random.randint(1, 20, size=10)
    a = np.random.rand(10) * 10 + 1
    n = np.random.randint(1, 100, size=10)
    pmf = [0.008076972, 2.950214e-05, 0.9799333, 3.216601e-06, 0.0003158895, 3.412497e-05, 4.350472e-10, 2.405773e-06, 5.860662e-06, 0.0001053948]
    cdf = [0.8964133, 0.9998666, 0.9799333, 0.9999995, 0.9998584, 0.9999458, 1.0, 0.999992, 0.9999977, 0.9998498]
    assert_allclose(zipfian.pmf(k, a, n)[1:], pmf[1:], rtol=1e-06)
    assert_allclose(zipfian.cdf(k, a, n)[1:], cdf[1:], rtol=5e-05)