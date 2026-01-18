import scipy.special as sc
import numpy as np
from numpy.testing import assert_equal, assert_allclose
def test_riemann_zeta_special_cases():
    assert np.isnan(sc.zeta(np.nan))
    assert sc.zeta(np.inf) == 1
    assert sc.zeta(0) == -0.5
    assert_equal(sc.zeta([-2, -4, -6, -8, -10]), 0)
    assert_allclose(sc.zeta(2), np.pi ** 2 / 6, rtol=1e-12)
    assert_allclose(sc.zeta(4), np.pi ** 4 / 90, rtol=1e-12)