import unittest
from numba.tests.support import TestCase
import ctypes
import operator
from functools import cached_property
import numpy as np
from numba import njit, types
from numba.extending import overload, intrinsic, overload_classmethod
from numba.core.target_extension import (
from numba.core import utils, fastmathpass, errors
from numba.core.dispatcher import Dispatcher
from numba.core.descriptors import TargetDescriptor
from numba.core import cpu, typing, cgutils
from numba.core.base import BaseContext
from numba.core.compiler_lock import global_compiler_lock
from numba.core import callconv
from numba.core.codegen import CPUCodegen, JITCodeLibrary
from numba.core.callwrapper import PyCallWrapper
from numba.core.imputils import RegistryLoader, Registry
from numba import _dynfunc
import llvmlite.binding as ll
from llvmlite import ir as llir
from numba.core.runtime import rtsys
from numba.core import compiler
from numba.core.compiler import CompilerBase, DefaultPassBuilder
from numba.core.compiler_machinery import FunctionPass, register_pass
from numba.core.typed_passes import PreLowerStripPhis
class TestTargetHierarchySelection(TestCase):
    """This tests that the target hierarchy is scanned in the right order,
    that appropriate functions are selected based on what's available and that
    the DPU target is distinctly different to the CPU"""

    def test_0_dpu_registry(self):
        """Checks that the DPU registry only contains the things added

        This test must be first to execute among all tests in this file to
        ensure the no lazily loaded entries are added yet.
        """
        self.assertFalse(dpu_function_registry.functions)
        self.assertFalse(dpu_function_registry.getattrs)
        self.assertEqual(len(dpu_function_registry.casts), 1)
        self.assertEqual(len(dpu_function_registry.constants), 3)

    def test_specialise_gpu(self):

        def my_func(x):
            pass

        @overload(my_func, target='generic')
        def ol_my_func1(x):

            def impl(x):
                return 1 + x
            return impl

        @overload(my_func, target='gpu')
        def ol_my_func2(x):

            def impl(x):
                return 10 + x
            return impl

        @djit()
        def dpu_foo():
            return my_func(7)

        @njit()
        def cpu_foo():
            return my_func(7)
        self.assertPreciseEqual(dpu_foo(), 3)
        self.assertPreciseEqual(cpu_foo(), 8)

    def test_specialise_dpu(self):

        def my_func(x):
            pass

        @overload(my_func, target='generic')
        def ol_my_func1(x):

            def impl(x):
                return 1 + x
            return impl

        @overload(my_func, target='gpu')
        def ol_my_func2(x):

            def impl(x):
                return 10 + x
            return impl

        @overload(my_func, target='dpu')
        def ol_my_func3(x):

            def impl(x):
                return 100 + x
            return impl

        @djit()
        def dpu_foo():
            return my_func(7)

        @njit()
        def cpu_foo():
            return my_func(7)
        self.assertPreciseEqual(dpu_foo(), 93)
        self.assertPreciseEqual(cpu_foo(), 8)

    def test_no_specialisation_found(self):

        def my_func(x):
            pass

        @overload(my_func, target='cuda')
        def ol_my_func_cuda(x):
            return lambda x: None

        @djit(nopython=True)
        def dpu_foo():
            my_func(1)
        accept = (errors.UnsupportedError, errors.TypingError)
        with self.assertRaises(accept) as raises:
            dpu_foo()
        msgs = ['Function resolution cannot find any matches for function', 'test_no_specialisation_found.<locals>.my_func', 'for the current target:', "'numba.tests.test_target_extension.DPU'"]
        for msg in msgs:
            self.assertIn(msg, str(raises.exception))

    def test_invalid_target_jit(self):
        with self.assertRaises(errors.NumbaValueError) as raises:

            @njit(_target='invalid_silicon')
            def foo():
                pass
            foo()
        msg = "No target is registered against 'invalid_silicon'"
        self.assertIn(msg, str(raises.exception))

    def test_invalid_target_overload(self):

        def bar():
            pass
        with self.assertRaises(errors.TypingError) as raises:

            @overload(bar, target='invalid_silicon')
            def ol_bar():
                return lambda: None

            @njit
            def foo():
                bar()
            foo()
        msg = "No target is registered against 'invalid_silicon'"
        self.assertIn(msg, str(raises.exception))

    def test_intrinsic_selection(self):
        """
        Test to make sure that targets can share generic implementations and
        cannot reach implementations that are not in their target hierarchy.
        """

        @intrinsic(target='generic')
        def intrin_math_generic(tyctx, x, y):
            sig = x(x, y)

            def codegen(cgctx, builder, tyargs, llargs):
                return builder.mul(*llargs)
            return (sig, codegen)

        @intrinsic(target='dpu')
        def intrin_math_dpu(tyctx, x, y):
            sig = x(x, y)

            def codegen(cgctx, builder, tyargs, llargs):
                return builder.sub(*llargs)
            return (sig, codegen)

        @intrinsic(target='cpu')
        def intrin_math_cpu(tyctx, x, y):
            sig = x(x, y)

            def codegen(cgctx, builder, tyargs, llargs):
                return builder.add(*llargs)
            return (sig, codegen)

        @njit
        def cpu_foo_specific():
            return intrin_math_cpu(3, 4)
        self.assertEqual(cpu_foo_specific(), 7)

        @njit
        def cpu_foo_generic():
            return intrin_math_generic(3, 4)
        self.assertEqual(cpu_foo_generic(), 12)

        @njit
        def cpu_foo_dpu():
            return intrin_math_dpu(3, 4)
        accept = (errors.UnsupportedError, errors.TypingError)
        with self.assertRaises(accept) as raises:
            cpu_foo_dpu()
        msgs = ['Function resolution cannot find any matches for function', 'intrinsic intrin_math_dpu', 'for the current target']
        for msg in msgs:
            self.assertIn(msg, str(raises.exception))

        @djit(nopython=True)
        def dpu_foo_specific():
            return intrin_math_dpu(3, 4)
        self.assertEqual(dpu_foo_specific(), -1)

        @djit(nopython=True)
        def dpu_foo_generic():
            return intrin_math_generic(3, 4)
        self.assertEqual(dpu_foo_generic(), 12)

        @djit(nopython=True)
        def dpu_foo_cpu():
            return intrin_math_cpu(3, 4)
        accept = (errors.UnsupportedError, errors.TypingError)
        with self.assertRaises(accept) as raises:
            dpu_foo_cpu()
        msgs = ['Function resolution cannot find any matches for function', 'intrinsic intrin_math_cpu', 'for the current target']
        for msg in msgs:
            self.assertIn(msg, str(raises.exception))

    def test_overload_allocation(self):

        def cast_integer(context, builder, val, fromty, toty):
            if toty.bitwidth == fromty.bitwidth:
                return val
            elif toty.bitwidth < fromty.bitwidth:
                return builder.trunc(val, context.get_value_type(toty))
            elif fromty.signed:
                return builder.sext(val, context.get_value_type(toty))
            else:
                return builder.zext(val, context.get_value_type(toty))

        @intrinsic(target='dpu')
        def intrin_alloc(typingctx, allocsize, align):
            """Intrinsic to call into the allocator for Array
            """

            def codegen(context, builder, signature, args):
                [allocsize, align] = args
                align_u32 = cast_integer(context, builder, align, signature.args[1], types.uint32)
                meminfo = context.nrt.meminfo_alloc_aligned(builder, allocsize, align_u32)
                return meminfo
            from numba.core.typing import signature
            mip = types.MemInfoPointer(types.voidptr)
            sig = signature(mip, allocsize, align)
            return (sig, codegen)

        @overload_classmethod(types.Array, '_allocate', target='dpu', jit_options={'nopython': True})
        def _ol_arr_allocate_dpu(cls, allocsize, align):

            def impl(cls, allocsize, align):
                return intrin_alloc(allocsize, align)
            return impl

        @overload(np.empty, target='dpu', jit_options={'nopython': True})
        def ol_empty_impl(n):

            def impl(n):
                return types.Array._allocate(n, 7)
            return impl

        def buffer_func():
            pass

        @overload(buffer_func, target='dpu', jit_options={'nopython': True})
        def ol_buffer_func_impl():

            def impl():
                return np.empty(10)
            return impl
        from numba.core.target_extension import target_override
        with target_override('dpu'):

            @djit(nopython=True)
            def foo():
                return buffer_func()
            r = foo()
        from numba.core.runtime import nrt
        self.assertIsInstance(r, nrt.MemInfo)