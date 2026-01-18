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
@pytest.mark.xfail(run=False)
def test_pcfd(self):

    def pcfd(v, x):
        return sc.pbdv(v, x)[0]
    assert_mpmath_equal(pcfd, exception_to_nan(lambda v, x: mpmath.pcfd(v, x, **HYPERKW)), [Arg(), Arg()])