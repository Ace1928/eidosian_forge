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
def test_set_random_state():
    rng1 = TransformedDensityRejection(StandardNormal(), random_state=123)
    rng2 = TransformedDensityRejection(StandardNormal())
    rng2.set_random_state(123)
    assert_equal(rng1.rvs(100), rng2.rvs(100))
    rng = TransformedDensityRejection(StandardNormal(), random_state=123)
    rvs1 = rng.rvs(100)
    rng.set_random_state(123)
    rvs2 = rng.rvs(100)
    assert_equal(rvs1, rvs2)