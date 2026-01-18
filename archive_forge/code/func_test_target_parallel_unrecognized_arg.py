import math
import numpy as np
from numba import int32, uint32, float32, float64, jit, vectorize
from numba.tests.support import tag, CheckWarningsMixin
import unittest
def test_target_parallel_unrecognized_arg(self):
    self._test_target_unrecognized_arg('parallel')