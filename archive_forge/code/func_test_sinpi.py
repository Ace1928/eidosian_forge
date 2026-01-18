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
def test_sinpi(self):
    eps = np.finfo(float).eps
    assert_mpmath_equal(_sinpi, mpmath.sinpi, [Arg()], nan_ok=False, rtol=2 * eps)