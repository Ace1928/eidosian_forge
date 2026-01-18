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
@pytest.mark.xfail(run=False, reason='Accuracy issues near z = -1 inherited from kv.')
def test_spherical_kn_complex(self):

    def mp_spherical_kn(n, z):
        arg = mpmath.mpmathify(z)
        out = mpmath.besselk(n + mpmath.mpf(1) / 2, arg) / mpmath.sqrt(2 * arg / mpmath.pi)
        if arg.imag == 0:
            return out.real
        else:
            return out
    assert_mpmath_equal(lambda n, z: sc.spherical_kn(int(n.real), z), exception_to_nan(mp_spherical_kn), [IntArg(0, 200), ComplexArg()], dps=200)