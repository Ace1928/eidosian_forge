import numpy as np
import ctypes
from numba.cuda.cudadrv.devicearray import (DeviceRecord, from_record_like,
from numba.cuda.testing import unittest, CUDATestCase
from numba.cuda.testing import skip_on_cudasim
from numba.np import numpy_support
from numba import cuda
def _check_device_record(self, reference, rec):
    self.assertEqual(rec.shape, tuple())
    self.assertEqual(rec.strides, tuple())
    self.assertEqual(rec.dtype, reference.dtype)
    self.assertEqual(rec.alloc_size, reference.dtype.itemsize)
    self.assertIsNotNone(rec.gpu_data)
    self.assertNotEqual(rec.device_ctypes_pointer, ctypes.c_void_p(0))
    numba_type = numpy_support.from_dtype(reference.dtype)
    self.assertEqual(rec._numba_type_, numba_type)