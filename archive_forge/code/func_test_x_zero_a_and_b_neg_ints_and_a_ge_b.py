import pytest
import numpy as np
from numpy.testing import assert_allclose, assert_equal
import scipy.special as sc
@pytest.mark.parametrize('a', [-3, -2])
def test_x_zero_a_and_b_neg_ints_and_a_ge_b(self, a):
    assert sc.hyp1f1(a, -3, 0) == 1