from numba import cuda, int32, float64, void
from numba.core.errors import TypingError
from numba.core import types
from numba.cuda.testing import unittest, CUDATestCase, skip_on_cudasim
import numpy as np
from numba.np import numpy_support as nps
from .extensions_usecases import test_struct_model_type, TestStruct
def _test_dynshared_slice(self, func, arr, expected):
    nshared = arr.size * arr.dtype.itemsize
    func[1, 1, 0, nshared](arr)
    np.testing.assert_array_equal(expected, arr)