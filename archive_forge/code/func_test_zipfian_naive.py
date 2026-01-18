import pytest
import itertools
from scipy.stats import (betabinom, betanbinom, hypergeom, nhypergeom,
import numpy as np
from numpy.testing import (
from scipy.special import binom as special_binom
from scipy.optimize import root_scalar
from scipy.integrate import quad
@pytest.mark.parametrize('a, n', naive_tests)
def test_zipfian_naive(self, a, n):

    @np.vectorize
    def Hns(n, s):
        """Naive implementation of harmonic sum"""
        return (1 / np.arange(1, n + 1) ** s).sum()

    @np.vectorize
    def pzip(k, a, n):
        """Naive implementation of zipfian pmf"""
        if k < 1 or k > n:
            return 0.0
        else:
            return 1 / k ** a / Hns(n, a)
    k = np.arange(n + 1)
    pmf = pzip(k, a, n)
    cdf = np.cumsum(pmf)
    mean = np.average(k, weights=pmf)
    var = np.average((k - mean) ** 2, weights=pmf)
    std = var ** 0.5
    skew = np.average(((k - mean) / std) ** 3, weights=pmf)
    kurtosis = np.average(((k - mean) / std) ** 4, weights=pmf) - 3
    assert_allclose(zipfian.pmf(k, a, n), pmf)
    assert_allclose(zipfian.cdf(k, a, n), cdf)
    assert_allclose(zipfian.stats(a, n, moments='mvsk'), [mean, var, skew, kurtosis])