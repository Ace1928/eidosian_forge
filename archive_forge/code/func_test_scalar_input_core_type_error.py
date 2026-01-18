import unittest
import pickle
import numpy as np
from numba import void, float32, float64, int32, int64, jit, guvectorize
from numba.np.ufunc import GUVectorize
from numba.tests.support import tag, TestCase
def test_scalar_input_core_type_error(self):
    with self.assertRaises(TypeError) as raises:

        @guvectorize(['int32[:], int32, int32[:]'], '(n),(n)->(n)', target=self.target)
        def pyfunc(a, b, c):
            pass
    self.assertEqual('scalar type int32 given for non scalar argument #2', str(raises.exception))