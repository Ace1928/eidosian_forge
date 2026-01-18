import sys
import hashlib
import pytest
import numpy as np
from numpy.linalg import LinAlgError
from numpy.testing import (
from numpy.random import Generator, MT19937, SeedSequence, RandomState
@pytest.mark.parametrize('mu', [-7.0, -np.pi, -3.1, np.pi, 3.2])
@pytest.mark.parametrize('kappa', [1e-09, 1e-06, 1, 1000.0, 1000000000000000.0])
def test_vonmises_large_kappa_range(self, mu, kappa):
    random = Generator(MT19937(self.seed))
    r = random.vonmises(mu, kappa, 50)
    assert_(np.all(r > -np.pi) and np.all(r <= np.pi))