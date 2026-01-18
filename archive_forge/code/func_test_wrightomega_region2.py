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
@pytest.mark.slow
@check_version(mpmath, '0.19')
def test_wrightomega_region2():
    x = np.linspace(-2, 1)
    y = np.linspace(-2 * np.pi, -1)
    x, y = np.meshgrid(x, y)
    z = (x + 1j * y).flatten()
    dataset = np.asarray([(z0, complex(_mpmath_wrightomega(z0, 25))) for z0 in z])
    FuncData(sc.wrightomega, dataset, 0, 1, rtol=1e-15).check()