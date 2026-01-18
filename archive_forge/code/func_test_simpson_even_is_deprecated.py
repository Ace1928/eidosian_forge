import pytest
import numpy as np
from numpy import cos, sin, pi
from numpy.testing import (assert_equal, assert_almost_equal, assert_allclose,
from scipy.integrate import (quadrature, romberg, romb, newton_cotes,
from scipy import stats, special as sc
def test_simpson_even_is_deprecated(self):
    x = np.linspace(0, 3, 4)
    y = x ** 2
    with pytest.deprecated_call():
        simpson(y, x=x, even='first')