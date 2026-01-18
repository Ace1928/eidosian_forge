import numpy as np
import ctypes
from numba.cuda.cudadrv.devicearray import (DeviceRecord, from_record_like,
from numba.cuda.testing import unittest, CUDATestCase
from numba.cuda.testing import skip_on_cudasim
from numba.np import numpy_support
from numba import cuda
def test_structured_array2(self):
    ary = self.samplerec1darr
    ary['g'] = 2
    ary['h'][0] = 3.0
    ary['h'][1] = 4.0
    self.assertEqual(ary['g'], 2)
    self.assertEqual(ary['h'][0], 3.0)
    self.assertEqual(ary['h'][1], 4.0)