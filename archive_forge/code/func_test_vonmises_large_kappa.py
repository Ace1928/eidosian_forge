import sys
import hashlib
import pytest
import numpy as np
from numpy.linalg import LinAlgError
from numpy.testing import (
from numpy.random import Generator, MT19937, SeedSequence, RandomState
@pytest.mark.parametrize('kappa', [10000.0, 1000000000000000.0])
def test_vonmises_large_kappa(self, kappa):
    random = Generator(MT19937(self.seed))
    rs = RandomState(random.bit_generator)
    state = random.bit_generator.state
    random_state_vals = rs.vonmises(0, kappa, size=10)
    random.bit_generator.state = state
    gen_vals = random.vonmises(0, kappa, size=10)
    if kappa < 1000000.0:
        assert_allclose(random_state_vals, gen_vals)
    else:
        assert np.all(random_state_vals != gen_vals)