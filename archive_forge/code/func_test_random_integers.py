import hashlib
import pickle
import sys
import warnings
import numpy as np
import pytest
from numpy.testing import (
from numpy.random import MT19937, PCG64
from numpy import random
def test_random_integers(self):
    random.seed(self.seed)
    with suppress_warnings() as sup:
        w = sup.record(DeprecationWarning)
        actual = random.random_integers(-99, 99, size=(3, 2))
        assert_(len(w) == 1)
    desired = np.array([[31, 3], [-52, 41], [-48, -66]])
    assert_array_equal(actual, desired)
    random.seed(self.seed)
    with suppress_warnings() as sup:
        w = sup.record(DeprecationWarning)
        actual = random.random_integers(198, size=(3, 2))
        assert_(len(w) == 1)
    assert_array_equal(actual, desired + 100)