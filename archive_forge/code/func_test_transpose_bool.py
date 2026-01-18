import numpy as np
from numba import cuda
from numba.cuda.kernels.transpose import transpose
from numba.cuda.testing import unittest
from numba.cuda.testing import skip_on_cudasim, CUDATestCase
def test_transpose_bool(self):
    for rows, cols in self.small_variants:
        with self.subTest(rows=rows, cols=cols):
            arr = np.random.randint(2, size=(rows, cols), dtype=np.bool_)
            transposed = arr.T
            d_arr = cuda.to_device(arr)
            d_transposed = cuda.device_array_like(transposed)
            transpose(d_arr, d_transposed)
            host_transposed = d_transposed.copy_to_host()
            np.testing.assert_array_equal(transposed, host_transposed)