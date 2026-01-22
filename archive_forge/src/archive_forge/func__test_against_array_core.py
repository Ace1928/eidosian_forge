import itertools
import numpy as np
from numba.cuda.cudadrv import devicearray
from numba import cuda
from numba.cuda.testing import unittest, CUDATestCase
from numba.cuda.testing import skip_on_cudasim
def _test_against_array_core(self, view):
    self.assertEqual(devicearray.is_contiguous(view), devicearray.array_core(view).flags['C_CONTIGUOUS'])