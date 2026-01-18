import numpy as np
from numba import cuda
from numba.core import types
from numba.cuda.testing import skip_on_cudasim, CUDATestCase
import unittest
from numba.np import numpy_support
@skip_on_cudasim('Will unexpectedly pass on cudasim')
@unittest.expectedFailure
def test_set_array(self):
    arr = np.zeros(2, dtype=recordwith2darray).view(np.recarray)
    rec = arr[0]
    nbarr = np.zeros(2, dtype=recordwith2darray).view(np.recarray)
    nbrec = nbarr[0]
    for pyfunc in (record_write_full_array, record_write_full_array_alt):
        pyfunc(rec)
        kernel = cuda.jit(pyfunc)
        kernel[1, 1](nbrec)
        np.testing.assert_equal(nbarr, arr)