import hashlib
import pickle
import sys
import warnings
import numpy as np
import pytest
from numpy.testing import (
from numpy.random import MT19937, PCG64
from numpy import random
def test_vonmises_large(self):
    random.seed(self.seed)
    actual = random.vonmises(mu=0.0, kappa=10000000.0, size=3)
    desired = np.array([0.0004634253748521111, 0.0003558873596114509, -0.0002337119622577433])
    assert_array_almost_equal(actual, desired, decimal=8)