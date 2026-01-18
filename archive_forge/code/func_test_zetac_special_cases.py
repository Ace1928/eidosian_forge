import scipy.special as sc
import numpy as np
from numpy.testing import assert_equal, assert_allclose
def test_zetac_special_cases():
    assert sc.zetac(np.inf) == 0
    assert np.isnan(sc.zetac(-np.inf))
    assert sc.zetac(0) == -1.5
    assert sc.zetac(1.0) == np.inf
    assert_equal(sc.zetac([-2, -50, -100]), -1)