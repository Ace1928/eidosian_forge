import pytest
import itertools
from scipy.stats import (betabinom, betanbinom, hypergeom, nhypergeom,
import numpy as np
from numpy.testing import (
from scipy.special import binom as special_binom
from scipy.optimize import root_scalar
from scipy.integrate import quad
def test_zipfian_continuity(self):
    alt1, agt1 = (0.99999999, 1.00000001)
    N = 30
    k = np.arange(1, N + 1)
    assert_allclose(zipfian.pmf(k, alt1, N), zipfian.pmf(k, agt1, N), rtol=5e-07)
    assert_allclose(zipfian.cdf(k, alt1, N), zipfian.cdf(k, agt1, N), rtol=5e-07)
    assert_allclose(zipfian.sf(k, alt1, N), zipfian.sf(k, agt1, N), rtol=5e-07)
    assert_allclose(zipfian.stats(alt1, N, moments='msvk'), zipfian.stats(agt1, N, moments='msvk'), rtol=5e-07)