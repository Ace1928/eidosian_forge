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
@pytest.mark.parametrize('u', u)
def test_ppf_hat(self, u):
    rng = TransformedDensityRejection(StandardNormal(), max_squeeze_hat_ratio=0.9999)
    with suppress_warnings() as sup:
        sup.filter(RuntimeWarning, 'invalid value encountered in greater')
        sup.filter(RuntimeWarning, 'invalid value encountered in greater_equal')
        sup.filter(RuntimeWarning, 'invalid value encountered in less')
        sup.filter(RuntimeWarning, 'invalid value encountered in less_equal')
        res = rng.ppf_hat(u)
        expected = stats.norm.ppf(u)
    assert_allclose(res, expected, rtol=0.001, atol=1e-05)
    assert res.shape == expected.shape