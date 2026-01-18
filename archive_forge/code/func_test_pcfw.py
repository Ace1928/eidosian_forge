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
def test_pcfw(self):

    def pcfw(a, x):
        return sc.pbwa(a, x)[0]

    def dpcfw(a, x):
        return sc.pbwa(a, x)[1]

    def mpmath_dpcfw(a, x):
        return mpmath.diff(mpmath.pcfw, (a, x), (0, 1))
    assert_mpmath_equal(pcfw, mpmath.pcfw, [Arg(-5, 5), Arg(-5, 5)], rtol=2e-08, n=100)
    assert_mpmath_equal(dpcfw, mpmath_dpcfw, [Arg(-5, 5), Arg(-5, 5)], rtol=2e-09, n=100)