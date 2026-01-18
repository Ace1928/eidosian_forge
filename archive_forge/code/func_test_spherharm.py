import numpy as np
from numpy.testing import assert_, assert_allclose
from numpy import pi
import pytest
import itertools
from scipy._lib import _pep440
import scipy.special as sc
from scipy.special._testutils import (
from scipy.special._mptestutils import (
from scipy.special._ufuncs import (
def test_spherharm(self):

    def spherharm(l, m, theta, phi):
        if m > l:
            return np.nan
        return sc.sph_harm(m, l, phi, theta)
    assert_mpmath_equal(spherharm, mpmath.spherharm, [IntArg(0, 100), IntArg(0, 100), Arg(a=0, b=pi), Arg(a=0, b=2 * pi)], atol=1e-08, n=6000, dps=150)