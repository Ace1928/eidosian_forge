import pytest
import itertools
from scipy.stats import (betabinom, betanbinom, hypergeom, nhypergeom,
import numpy as np
from numpy.testing import (
from scipy.special import binom as special_binom
from scipy.optimize import root_scalar
from scipy.integrate import quad
def test_skellam_gh11474():
    mu = [1, 10, 100, 1000, 5000, 5050, 5100, 5250, 6000]
    cdf = skellam.cdf(0, mu, mu)
    cdf_expected = [0.6542541612768356, 0.5448901559424127, 0.514113579974558, 0.5044605891382528, 0.501994736335045, 0.5019848365953181, 0.5019750827993392, 0.501946662180506, 0.5018209330219539]
    assert_allclose(cdf, cdf_expected)