import math
import numpy as np
from numba import cuda
from numba.cuda.testing import unittest
from numba.cuda.testing import skip_on_cudasim, CUDATestCase
from numba.cuda.random import \
@skip_on_cudasim('skip test for speed under cudasim')
def test_normal_float64(self):
    self.check_normal(rng_kernel_float64, np.float64)