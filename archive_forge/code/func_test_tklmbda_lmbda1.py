import itertools
import numpy as np
from numpy.testing import assert_equal, assert_allclose
import pytest
import scipy.special as sp
from scipy.special._testutils import (
from scipy.special._mptestutils import (
@pytest.mark.parametrize('lmbda', [0.5, 1.0, 8.0])
def test_tklmbda_lmbda1(self, lmbda):
    bound = 1 / lmbda
    assert_equal(sp.tklmbda([-bound, bound], lmbda), [0.0, 1.0])