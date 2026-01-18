import threading
import pickle
import pytest
from copy import deepcopy
import platform
import sys
import math
import numpy as np
from numpy.testing import assert_allclose, assert_equal, suppress_warnings
from scipy.stats.sampling import (
from pytest import raises as assert_raises
from scipy import stats
from scipy import special
from scipy.stats import chisquare, cramervonmises
from scipy.stats._distr_params import distdiscrete, distcont
from scipy._lib._util import check_random_state
def test_with_scipy_distribution():
    dist = stats.norm()
    urng = np.random.default_rng(0)
    rng = NumericalInverseHermite(dist, random_state=urng)
    u = np.linspace(0, 1, num=100)
    check_cont_samples(rng, dist, dist.stats())
    assert_allclose(dist.ppf(u), rng.ppf(u))
    dist = stats.norm(loc=10.0, scale=5.0)
    rng = NumericalInverseHermite(dist, random_state=urng)
    check_cont_samples(rng, dist, dist.stats())
    assert_allclose(dist.ppf(u), rng.ppf(u))
    dist = stats.binom(10, 0.2)
    rng = DiscreteAliasUrn(dist, random_state=urng)
    domain = dist.support()
    pv = dist.pmf(np.arange(domain[0], domain[1] + 1))
    check_discr_samples(rng, pv, dist.stats())