import numpy as np
import ctypes
from numba import jit, literal_unroll, njit, typeof
from numba.core import types
from numba.core.itanium_mangler import mangle_type
from numba.core.errors import TypingError
import unittest
from numba.np import numpy_support
from numba.tests.support import TestCase, skip_ppc64le_issue6465
def test_structure_dtype_with_titles(self):
    vecint4 = np.dtype([(('x', 's0'), 'i4'), (('y', 's1'), 'i4'), (('z', 's2'), 'i4'), (('w', 's3'), 'i4')])
    nbtype = numpy_support.from_dtype(vecint4)
    self.assertEqual(len(nbtype.fields), len(vecint4.fields))
    arr = np.zeros(10, dtype=vecint4)

    def pyfunc(a):
        for i in range(a.size):
            j = i + 1
            a[i]['s0'] = j * 2
            a[i]['x'] += -1
            a[i]['s1'] = j * 3
            a[i]['y'] += -2
            a[i]['s2'] = j * 4
            a[i]['z'] += -3
            a[i]['s3'] = j * 5
            a[i]['w'] += -4
        return a
    expect = pyfunc(arr.copy())
    cfunc = self.get_cfunc(pyfunc, (nbtype[:],))
    got = cfunc(arr.copy())
    np.testing.assert_equal(expect, got)