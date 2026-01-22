import inspect
import math
import operator
import sys
import pickle
import multiprocessing
import ctypes
import warnings
import re
import numpy as np
from llvmlite import ir
import numba
from numba import njit, jit, vectorize, guvectorize, objmode
from numba.core import types, errors, typing, compiler, cgutils
from numba.core.typed_passes import type_inference_stage
from numba.core.registry import cpu_target
from numba.core.imputils import lower_constant
from numba.tests.support import (
from numba.core.errors import LoweringError
import unittest
from numba.extending import (
from numba.core.typing.templates import (
from .pdlike_usecase import Index, Series
class TestLowLevelExtending(TestCase):
    """
    Test the low-level two-tier extension API.
    """

    def test_func1(self):
        pyfunc = call_func1_nullary
        cfunc = jit(nopython=True)(pyfunc)
        self.assertPreciseEqual(cfunc(), 42)
        pyfunc = call_func1_unary
        cfunc = jit(nopython=True)(pyfunc)
        self.assertPreciseEqual(cfunc(None), 42)
        self.assertPreciseEqual(cfunc(18.0), 6.0)

    @TestCase.run_test_in_subprocess
    def test_func1_isolated(self):
        self.test_func1()

    def test_type_callable_keeps_function(self):
        self.assertIs(type_func1, type_func1_)
        self.assertIsNotNone(type_func1)

    @TestCase.run_test_in_subprocess
    def test_cast_mydummy(self):
        pyfunc = get_dummy
        cfunc = njit(types.float64())(pyfunc)
        self.assertPreciseEqual(cfunc(), 42.0)

    def test_mk_func_literal(self):
        """make sure make_function is passed to typer class as a literal
        """
        test_ir = compiler.run_frontend(mk_func_test_impl)
        typingctx = cpu_target.typing_context
        targetctx = cpu_target.target_context
        typingctx.refresh()
        targetctx.refresh()
        typing_res = type_inference_stage(typingctx, targetctx, test_ir, (), None)
        self.assertTrue(any((isinstance(a, types.MakeFunctionLiteral) for a in typing_res.typemap.values())))