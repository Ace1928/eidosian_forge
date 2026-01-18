from numpy import float32, float64, complex64, complex128, arange, array, \
from scipy.linalg import _fblas as fblas
from numpy.testing import assert_array_equal, \
import pytest
def test_sgemv_on_osx(self):
    from itertools import product
    import sys
    import numpy as np
    if sys.platform != 'darwin':
        return

    def aligned_array(shape, align, dtype, order='C'):
        d = dtype()
        N = np.prod(shape)
        tmp = np.zeros(N * d.nbytes + align, dtype=np.uint8)
        address = tmp.__array_interface__['data'][0]
        for offset in range(align):
            if (address + offset) % align == 0:
                break
        tmp = tmp[offset:offset + N * d.nbytes].view(dtype=dtype)
        return tmp.reshape(shape, order=order)

    def as_aligned(arr, align, dtype, order='C'):
        aligned = aligned_array(arr.shape, align, dtype, order)
        aligned[:] = arr[:]
        return aligned

    def assert_dot_close(A, X, desired):
        assert_allclose(self.blas_func(1.0, A, X), desired, rtol=1e-05, atol=1e-07)
    testdata = product((15, 32), (10000,), (200, 89), ('C', 'F'))
    for align, m, n, a_order in testdata:
        A_d = np.random.rand(m, n)
        X_d = np.random.rand(n)
        desired = np.dot(A_d, X_d)
        A_f = as_aligned(A_d, align, np.float32, order=a_order)
        X_f = as_aligned(X_d, align, np.float32, order=a_order)
        assert_dot_close(A_f, X_f, desired)