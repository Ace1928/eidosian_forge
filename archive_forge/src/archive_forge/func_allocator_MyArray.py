import builtins
import unittest
from numbers import Number
from functools import wraps
import numpy as np
from llvmlite import ir
import numba
from numba import njit, typeof, objmode
from numba.core import cgutils, types, typing
from numba.core.pythonapi import box
from numba.core.errors import TypingError
from numba.core.registry import cpu_target
from numba.extending import (intrinsic, lower_builtin, overload_classmethod,
from numba.np import numpy_support
from numba.tests.support import TestCase, MemoryLeakMixin
@intrinsic
def allocator_MyArray(typingctx, allocsize, align):

    def impl(context, builder, sig, args):
        context.nrt._require_nrt()
        size, align = args
        mod = builder.module
        u32 = ir.IntType(32)
        voidptr = cgutils.voidptr_t
        get_alloc_fnty = ir.FunctionType(voidptr, ())
        get_alloc_fn = cgutils.get_or_insert_function(mod, get_alloc_fnty, name='_nrt_get_sample_external_allocator')
        ext_alloc = builder.call(get_alloc_fn, ())
        fnty = ir.FunctionType(voidptr, [cgutils.intp_t, u32, voidptr])
        fn = cgutils.get_or_insert_function(mod, fnty, name='NRT_MemInfo_alloc_safe_aligned_external')
        fn.return_value.add_attribute('noalias')
        if isinstance(align, builtins.int):
            align = context.get_constant(types.uint32, align)
        else:
            assert align.type == u32, 'align must be a uint32'
        call = builder.call(fn, [size, align, ext_alloc])
        call.name = 'allocate_MyArray'
        return call
    mip = types.MemInfoPointer(types.voidptr)
    sig = typing.signature(mip, allocsize, align)
    return (sig, impl)