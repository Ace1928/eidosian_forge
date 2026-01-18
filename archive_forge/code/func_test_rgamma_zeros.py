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
@check_version(mpmath, '0.19')
@pytest.mark.slow
def test_rgamma_zeros():
    dx = np.r_[-np.logspace(-1, -13, 3), 0, np.logspace(-13, -1, 3)]
    dy = dx.copy()
    dx, dy = np.meshgrid(dx, dy)
    dz = dx + 1j * dy
    zeros = np.arange(0, -170, -1).reshape(1, 1, -1)
    z = (zeros + np.dstack((dz,) * zeros.size)).flatten()
    with mpmath.workdps(100):
        dataset = [(z0, complex(mpmath.rgamma(z0))) for z0 in z]
    dataset = np.array(dataset)
    FuncData(sc.rgamma, dataset, 0, 1, rtol=1e-12).check()