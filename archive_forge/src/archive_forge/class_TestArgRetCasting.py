import os, sys, subprocess
import dis
import itertools
import numpy as np
import numba
from numba import jit, njit
from numba.core import errors, ir, types, typing, typeinfer, utils
from numba.core.typeconv import Conversion
from numba.extending import overload_method
from numba.tests.support import TestCase, tag
from numba.tests.test_typeconv import CompatibilityTestMixin
from numba.core.untyped_passes import TranslateByteCode, IRProcessing
from numba.core.typed_passes import PartialTypeInference
from numba.core.compiler_machinery import FunctionPass, register_pass
import unittest
class TestArgRetCasting(unittest.TestCase):

    def test_arg_ret_casting(self):

        def foo(x):
            return x
        args = (i32,)
        return_type = f32
        cfunc = njit(return_type(*args))(foo)
        cres = cfunc.overloads[args]
        self.assertTrue(isinstance(cfunc(123), float))
        self.assertEqual(cres.signature.args, args)
        self.assertEqual(cres.signature.return_type, return_type)

    def test_arg_ret_mismatch(self):

        def foo(x):
            return x
        args = (types.Array(i32, 1, 'C'),)
        return_type = f32
        try:
            njit(return_type(*args))(foo)
        except errors.TypingError as e:
            pass
        else:
            self.fail('Should complain about array casting to float32')

    def test_invalid_arg_type_forcing(self):

        def foo(iters):
            a = range(iters)
            return iters
        args = (u32,)
        return_type = u8
        cfunc = njit(return_type(*args))(foo)
        cres = cfunc.overloads[args]
        typemap = cres.type_annotation.typemap
        self.assertEqual(typemap['iters'], u32)