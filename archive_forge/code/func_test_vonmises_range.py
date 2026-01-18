from numpy.testing import (assert_, assert_array_equal)
import numpy as np
import pytest
from numpy.random import Generator, MT19937
def test_vonmises_range(self):
    for mu in np.linspace(-7.0, 7.0, 5):
        r = self.mt19937.vonmises(mu, 1, 50)
        assert_(np.all(r > -np.pi) and np.all(r <= np.pi))