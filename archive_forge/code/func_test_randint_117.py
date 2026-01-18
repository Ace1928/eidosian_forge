import sys
import pytest
from numpy.testing import (
import numpy as np
from numpy import random
@pytest.mark.skipif(np.iinfo('l').max < 2 ** 32, reason='Cannot test with 32-bit C long')
def test_randint_117(self):
    random.seed(0)
    expected = np.array([2357136044, 2546248239, 3071714933, 3626093760, 2588848963, 3684848379, 2340255427, 3638918503, 1819583497, 2678185683], dtype='int64')
    actual = random.randint(2 ** 32, size=10)
    assert_array_equal(actual, expected)