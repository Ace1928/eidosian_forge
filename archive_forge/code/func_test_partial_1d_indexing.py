import decimal
import itertools
import numpy as np
import unittest
from numba import jit, njit, typeof
from numba.core import utils, types, errors
from numba.tests.support import TestCase, tag
from numba.core.typing import arraydecl
from numba.core.types import intp, ellipsis, slice2_type, slice3_type
def test_partial_1d_indexing(self, flags=enable_pyobj_flags):
    pyfunc = partial_1d_usecase

    def check(arr, arraytype):
        argtys = (arraytype, types.int32)
        cfunc = jit(argtys, **flags)(pyfunc)
        self.assertEqual(pyfunc(arr, 0), cfunc(arr, 0))
        n = arr.shape[0] - 1
        self.assertEqual(pyfunc(arr, n), cfunc(arr, n))
        self.assertEqual(pyfunc(arr, -1), cfunc(arr, -1))
    a = np.arange(12, dtype='i4').reshape((4, 3))
    arraytype = types.Array(types.int32, 2, 'C')
    check(a, arraytype)
    a = np.arange(12, dtype='i4').reshape((3, 4)).T
    arraytype = types.Array(types.int32, 2, 'F')
    check(a, arraytype)
    a = np.arange(12, dtype='i4').reshape((3, 4))[::2]
    arraytype = types.Array(types.int32, 2, 'A')
    check(a, arraytype)