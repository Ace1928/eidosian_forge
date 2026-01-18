import sys
import hashlib
import pytest
import numpy as np
from numpy.linalg import LinAlgError
from numpy.testing import (
from numpy.random import Generator, MT19937, SeedSequence, RandomState
def test_repeatability2(self):
    random = Generator(MT19937(self.seed))
    sample = random.multivariate_hypergeometric([20, 30, 50], 50, size=5, method='marginals')
    expected = np.array([[9, 17, 24], [7, 13, 30], [9, 15, 26], [9, 17, 24], [12, 14, 24]])
    assert_array_equal(sample, expected)