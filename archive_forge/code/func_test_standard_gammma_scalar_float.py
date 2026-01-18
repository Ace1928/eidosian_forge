import sys
import hashlib
import pytest
import numpy as np
from numpy.linalg import LinAlgError
from numpy.testing import (
from numpy.random import Generator, MT19937, SeedSequence, RandomState
def test_standard_gammma_scalar_float(self):
    random = Generator(MT19937(self.seed))
    actual = random.standard_gamma(3, dtype=np.float32)
    desired = 2.9242148399353027
    assert_array_almost_equal(actual, desired, decimal=6)