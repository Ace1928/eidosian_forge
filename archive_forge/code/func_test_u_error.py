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
def test_u_error(self):
    dist = StandardNormal()
    rng = NumericalInverseHermite(dist, u_resolution=1e-10)
    max_error, mae = rng.u_error()
    assert max_error < 1e-10
    assert mae <= max_error
    with suppress_warnings() as sup:
        sup.filter(RuntimeWarning)
        rng = NumericalInverseHermite(dist, u_resolution=1e-14)
    max_error, mae = rng.u_error()
    assert max_error < 1e-14
    assert mae <= max_error