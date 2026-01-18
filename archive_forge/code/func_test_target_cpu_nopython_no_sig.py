import math
import numpy as np
from numba import int32, uint32, float32, float64, jit, vectorize
from numba.tests.support import tag, CheckWarningsMixin
import unittest
def test_target_cpu_nopython_no_sig(self):
    self._test_target_nopython('cpu', [], False)