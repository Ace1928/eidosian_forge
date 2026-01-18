import contextlib
import sys
import numpy as np
from numba import vectorize, guvectorize
from numba.tests.support import (TestCase, CheckWarningsMixin,
import unittest
def test_truediv_float(self):
    self.check_truediv_real(np.float64)