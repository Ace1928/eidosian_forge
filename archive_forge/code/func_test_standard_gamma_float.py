import sys
import hashlib
import pytest
import numpy as np
from numpy.linalg import LinAlgError
from numpy.testing import (
from numpy.random import Generator, MT19937, SeedSequence, RandomState
def test_standard_gamma_float(self):
    random = Generator(MT19937(self.seed))
    actual = random.standard_gamma(shape=3, size=(3, 2))
    desired = np.array([[0.62971, 1.2238], [3.89941, 4.1248], [3.74994, 3.74929]])
    assert_array_almost_equal(actual, desired, decimal=5)