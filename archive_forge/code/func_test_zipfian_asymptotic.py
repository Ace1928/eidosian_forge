import pytest
import itertools
from scipy.stats import (betabinom, betanbinom, hypergeom, nhypergeom,
import numpy as np
from numpy.testing import (
from scipy.special import binom as special_binom
from scipy.optimize import root_scalar
from scipy.integrate import quad
def test_zipfian_asymptotic(self):
    a = 6.5
    N = 10000000
    k = np.arange(1, 21)
    assert_allclose(zipfian.pmf(k, a, N), zipf.pmf(k, a))
    assert_allclose(zipfian.cdf(k, a, N), zipf.cdf(k, a))
    assert_allclose(zipfian.sf(k, a, N), zipf.sf(k, a))
    assert_allclose(zipfian.stats(a, N, moments='msvk'), zipf.stats(a, moments='msvk'))