import pytest
import numpy as np
from numpy.testing import assert_allclose, assert_equal
import scipy.special as sc
def test_poles(self):
    assert_equal(sc.hyp1f1(1, [0, -1, -2, -3, -4], 0.5), np.inf)