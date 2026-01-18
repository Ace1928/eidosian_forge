import numpy as np
from numba import cuda, vectorize, guvectorize
from numba.np.numpy_support import from_dtype
from numba.cuda.testing import CUDATestCase, skip_on_cudasim
import unittest
@skip_on_cudasim('no .copy_to_host() in the simulator')
def test_timedelta_view_as_int64(self):
    arr = np.arange('2005-02', '2006-02', dtype='datetime64[D]')
    arr = arr - (arr - 1)
    self.assertEqual(arr.dtype, np.dtype('timedelta64[D]'))
    darr = cuda.to_device(arr)
    viewed = darr.view(np.int64)
    self.assertPreciseEqual(arr.view(np.int64), viewed.copy_to_host())
    self.assertEqual(viewed.gpu_data, darr.gpu_data)