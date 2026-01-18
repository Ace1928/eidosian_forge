import numpy as np
import ctypes
from numba import jit, literal_unroll, njit, typeof
from numba.core import types
from numba.core.itanium_mangler import mangle_type
from numba.core.errors import TypingError
import unittest
from numba.np import numpy_support
from numba.tests.support import TestCase, skip_ppc64le_issue6465
def test_record_return(self):
    pyfunc = record_return
    recty = numpy_support.from_dtype(recordtype)
    cfunc = self.get_cfunc(pyfunc, (recty[:], types.intp))
    attrs = 'abc'
    indices = [0, 1, 2]
    for index, attr in zip(indices, attrs):
        nbary = self.nbsample1d.copy()
        with self.assertRefCount(nbary):
            res = cfunc(nbary, index)
            self.assertEqual(nbary[index], res)
            setattr(res, attr, 0)
            self.assertNotEqual(nbary[index], res)
            del res