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
@pytest.mark.xfail(reason='see gh-3551 for bad points on 32 bit systems and gh-8095 for another bad point')
def test_rf(self):
    if _pep440.parse(mpmath.__version__) >= _pep440.Version('1.0.0'):
        mppoch = mpmath.rf
    else:

        def mppoch(a, m):
            if float(a + m) == int(a + m) and float(a + m) <= 0:
                a = mpmath.mpf(a)
                m = int(a + m) - a
            return mpmath.rf(a, m)
    assert_mpmath_equal(sc.poch, mppoch, [Arg(), Arg()], dps=400)