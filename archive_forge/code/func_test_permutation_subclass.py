import sys
import pytest
from numpy.testing import (
import numpy as np
from numpy import random
def test_permutation_subclass(self):

    class N(np.ndarray):
        pass
    random.seed(1)
    orig = np.arange(3).view(N)
    perm = random.permutation(orig)
    assert_array_equal(perm, np.array([0, 2, 1]))
    assert_array_equal(orig, np.arange(3).view(N))

    class M:
        a = np.arange(5)

        def __array__(self):
            return self.a
    random.seed(1)
    m = M()
    perm = random.permutation(m)
    assert_array_equal(perm, np.array([2, 1, 4, 0, 3]))
    assert_array_equal(m.__array__(), np.arange(5))