import random
import numpy as np
from numba.tests.support import TestCase, captured_stdout
from numba import njit, literally
from numba.core import types
from numba.cpython.unsafe.tuple import tuple_setitem, build_full_slice_tuple
from numba.np.unsafe.ndarray import to_fixed_tuple, empty_inferred
from numba.core.unsafe.bytes import memcpy_region
from numba.core.unsafe.refcount import dump_refcount
from numba.cpython.unsafe.numbers import trailing_zeros, leading_zeros
from numba.core.errors import TypingError
class TestRefCount(TestCase):

    def test_dump_refcount(self):

        @njit
        def use_dump_refcount():
            a = np.ones(10)
            b = (a, a)
            dump_refcount(a)
            dump_refcount(b)
        with captured_stdout() as stream:
            use_dump_refcount()
        output = stream.getvalue()
        pat = 'dump refct of {}'
        aryty = types.float64[::1]
        tupty = types.Tuple.from_types([aryty] * 2)
        self.assertIn(pat.format(aryty), output)
        self.assertIn(pat.format(tupty), output)