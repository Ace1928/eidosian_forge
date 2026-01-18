import array
from collections import namedtuple
import enum
import mmap
import typing as py_typing
import numpy as np
import unittest
from numba.core import types
from numba.core.errors import NumbaValueError, NumbaTypeError
from numba.misc.special import typeof
from numba.core.dispatcher import OmittedArg
from numba._dispatcher import compute_fingerprint
from numba.tests.support import TestCase, skip_unless_cffi, tag
from numba.tests.test_numpy_support import ValueTypingTestBase
from numba.tests.ctypes_usecases import *
from numba.tests.enum_usecases import *
from numba.np import numpy_support
def test_structured_arrays(self):

    def check(arr, dtype, ndim, layout, aligned):
        ty = typeof(arr)
        self.assertIsInstance(ty, types.Array)
        self.assertEqual(ty.dtype, dtype)
        self.assertEqual(ty.ndim, ndim)
        self.assertEqual(ty.layout, layout)
        self.assertEqual(ty.aligned, aligned)
    dtype = np.dtype([('m', np.int32), ('n', 'S5')])
    rec_ty = numpy_support.from_struct_dtype(dtype)
    arr = np.empty(4, dtype=dtype)
    check(arr, rec_ty, 1, 'C', False)
    arr = np.recarray(4, dtype=dtype)
    check(arr, rec_ty, 1, 'C', False)
    dtype = np.dtype([('m', np.int32), ('n', 'S5')], align=True)
    rec_ty = numpy_support.from_struct_dtype(dtype)
    arr = np.empty(4, dtype=dtype)
    check(arr, rec_ty, 1, 'C', True)
    arr = np.recarray(4, dtype=dtype)
    check(arr, rec_ty, 1, 'C', True)