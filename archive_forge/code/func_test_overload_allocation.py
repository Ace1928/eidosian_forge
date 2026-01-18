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