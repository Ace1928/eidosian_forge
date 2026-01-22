from numba import vectorize, jit, bool_, double, int_, float_, typeof, int8
import unittest
import numpy as np
class TestVectTypeInfer(unittest.TestCase):

    def test_type_inference(self):
        """This is testing numpy ufunc dispatch machinery
        """
        global vector_add
        vector_add = vectorize([bool_(double, int_), double(double, double), float_(double, float_)])(add)

        def numba_type_equal(a, b):
            self.assertEqual(a.dtype, b.dtype)
            self.assertEqual(a.ndim, b.ndim)
        numba_type_equal(func(np.dtype(np.float64), np.dtype('i')), bool_[:])
        numba_type_equal(func(np.dtype(np.float64), np.dtype(np.float64)), double[:])
        numba_type_equal(func(np.dtype(np.float64), np.dtype(np.float32)), double[:])