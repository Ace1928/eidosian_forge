import pytest
import numpy as np
from numpy.testing import assert_equal, assert_allclose
import scipy.special as sc
from scipy.special import rgamma, wright_bessel
@pytest.mark.parametrize('a', [0, 1e-06, 0.1, 0.5, 1, 10])
@pytest.mark.parametrize('b', [1, 1 + 0.001, 2, 5, 10])
@pytest.mark.parametrize('x', [0, 1e-06, 0.1, 0.5, 1, 5, 10, 100])
def test_wright_functional(a, b, x):
    """Test functional relation of wright_bessel.

    Phi(a, b-1, z) = a*z*Phi(a, b+a, z) + (b-1)*Phi(a, b, z)

    Note that d/dx Phi(a, b, x) = Phi(a, b-1, x)
    See Eq. (22) of
    B. Stankovic, On the Function of E. M. Wright,
    Publ. de l' Institut Mathematique, Beograd,
    Nouvelle S`er. 10 (1970), 113-124.
    """
    assert_allclose(wright_bessel(a, b - 1, x), a * x * wright_bessel(a, b + a, x) + (b - 1) * wright_bessel(a, b, x), rtol=1e-08, atol=1e-08)