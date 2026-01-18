import numpy as np
from numpy.testing import assert_equal, assert_allclose
import scipy.special as sc
def test_outside_of_domain(self):
    assert all(np.isnan(sc.ndtri([-1.5, 1.5])))