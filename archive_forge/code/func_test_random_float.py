import sys
import hashlib
import pytest
import numpy as np
from numpy.linalg import LinAlgError
from numpy.testing import (
from numpy.random import Generator, MT19937, SeedSequence, RandomState
def test_random_float(self):
    random = Generator(MT19937(self.seed))
    actual = random.random((3, 2))
    desired = np.array([[0.0969992, 0.70751746], [0.08436483, 0.76773121], [0.66506902, 0.71548719]])
    assert_array_almost_equal(actual, desired, decimal=7)