from numpy.testing import assert_equal, assert_allclose
import scipy.constants as sc
def test_nu_to_lambda():
    assert_equal(sc.nu2lambda([sc.speed_of_light, 1]), [1, sc.speed_of_light])