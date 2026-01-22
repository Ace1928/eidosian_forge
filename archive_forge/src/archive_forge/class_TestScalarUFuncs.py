import functools
import itertools
import sys
import warnings
import threading
import operator
import numpy as np
import unittest
from numba import guvectorize, njit, typeof, vectorize
from numba.core import types
from numba.np.numpy_support import from_dtype
from numba.core.errors import LoweringError, TypingError
from numba.tests.support import TestCase, MemoryLeakMixin
from numba.core.typing.npydecl import supported_ufuncs
from numba.np import numpy_support
from numba.core.registry import cpu_target
from numba.core.base import BaseContext
from numba.np import ufunc_db
class TestScalarUFuncs(TestCase):
    """check the machinery of ufuncs works when the result is an scalar.
    These are not exhaustive because:
    - the machinery to support this case is the same for all the functions of a
      given arity.
    - the result of the inner function itself is already tested in TestUFuncs
    """

    def run_ufunc(self, pyfunc, arg_types, arg_values):
        for tyargs, args in zip(arg_types, arg_values):
            cfunc = njit(tyargs)(pyfunc)
            got = cfunc(*args)
            expected = pyfunc(*_as_dtype_value(tyargs, args))
            msg = 'for args {0} typed {1}'.format(args, tyargs)
            special = set([(types.int32, types.uint64), (types.uint64, types.int32), (types.int64, types.uint64), (types.uint64, types.int64)])
            if tyargs in special:
                expected = float(expected)
            elif np.issubdtype(expected.dtype, np.inexact):
                expected = float(expected)
            elif np.issubdtype(expected.dtype, np.integer):
                expected = int(expected)
            elif np.issubdtype(expected.dtype, np.bool_):
                expected = bool(expected)
            alltypes = tyargs + (cfunc.overloads[tyargs].signature.return_type,)
            if any([t == types.float32 for t in alltypes]):
                prec = 'single'
            elif any([t == types.float64 for t in alltypes]):
                prec = 'double'
            else:
                prec = 'exact'
            self.assertPreciseEqual(got, expected, msg=msg, prec=prec)

    def test_scalar_unary_ufunc(self):

        def _func(x):
            return np.sqrt(x)
        vals = [(2,), (2,), (1,), (2,), (0.1,), (0.2,)]
        tys = [(types.int32,), (types.uint32,), (types.int64,), (types.uint64,), (types.float32,), (types.float64,)]
        self.run_ufunc(_func, tys, vals)

    def test_scalar_binary_uniform_ufunc(self):

        def _func(x, y):
            return np.add(x, y)
        vals = [2, 2, 1, 2, 0.1, 0.2]
        tys = [types.int32, types.uint32, types.int64, types.uint64, types.float32, types.float64]
        self.run_ufunc(_func, zip(tys, tys), zip(vals, vals))

    def test_scalar_binary_mixed_ufunc(self):

        def _func(x, y):
            return np.add(x, y)
        vals = [2, 2, 1, 2, 0.1, 0.2]
        tys = [types.int32, types.uint32, types.int64, types.uint64, types.float32, types.float64]
        self.run_ufunc(_func, itertools.product(tys, tys), itertools.product(vals, vals))