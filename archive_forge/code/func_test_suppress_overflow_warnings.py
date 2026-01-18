import warnings
import sys
import os
import itertools
import pytest
import weakref
import numpy as np
from numpy.testing import (
def test_suppress_overflow_warnings(self):
    with pytest.raises(AssertionError):
        with np.errstate(all='raise'):
            np.testing.assert_array_equal(np.array([1, 2, 3], np.float32), np.array([1, 1e-40, 3], np.float32))