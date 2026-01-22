from itertools import product, combinations_with_replacement
import numpy as np
from numba import jit, njit, typeof
from numba.np.numpy_support import numpy_version
from numba.tests.support import TestCase, MemoryLeakMixin, tag
import unittest
class TestArrayReductionsExceptions(MemoryLeakMixin, TestCase):
    zero_size = np.arange(0)

    def check_exception(self, pyfunc, msg):
        cfunc = jit(nopython=True)(pyfunc)
        with self.assertRaises(BaseException):
            pyfunc(self.zero_size)
        with self.assertRaises(ValueError) as e:
            cfunc(self.zero_size)
        self.assertIn(msg, str(e.exception))

    @classmethod
    def install(cls):
        fn_to_msg = dict()
        empty_seq = 'attempt to get {0} of an empty sequence'
        op_no_ident = 'zero-size array to reduction operation {0}'
        for x in [array_argmax, array_argmax_global, array_argmin, array_argmin_global]:
            fn_to_msg[x] = empty_seq
        for x in [array_max, array_max, array_min, array_min]:
            fn_to_msg[x] = op_no_ident
        name_template = 'test_zero_size_array_{0}'
        for fn, msg in fn_to_msg.items():
            test_name = name_template.format(fn.__name__)
            lmsg = msg.format(fn.__name__)
            lmsg = lmsg.replace('array_', '').replace('_global', '')

            def test_fn(self, func=fn, message=lmsg):
                self.check_exception(func, message)
            setattr(cls, test_name, test_fn)