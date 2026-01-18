import numpy as np
from scipy import linalg
from statsmodels.tools.decorators import cache_readonly
def testcompare(m1, m2):
    from numpy.testing import assert_almost_equal, assert_approx_equal
    decimal = 12
    assert_almost_equal(m1.minv, m2.minv, decimal=decimal)
    s1 = np.sign(m1.mhalf.sum(1))[:, None]
    s2 = np.sign(m2.mhalf.sum(1))[:, None]
    scorr = s1 / s2
    assert_almost_equal(m1.mhalf, m2.mhalf * scorr, decimal=decimal)
    assert_almost_equal(m1.minvhalf, m2.minvhalf, decimal=decimal)
    evals1, evecs1 = m1.meigh
    evals2, evecs2 = m2.meigh
    assert_almost_equal(evals1, evals2, decimal=decimal)
    s1 = np.sign(evecs1.sum(0))
    s2 = np.sign(evecs2.sum(0))
    scorr = s1 / s2
    assert_almost_equal(evecs1, evecs2 * scorr, decimal=decimal)
    assert_approx_equal(m1.mdet, m2.mdet, significant=13)
    assert_approx_equal(m1.mlogdet, m2.mlogdet, significant=13)