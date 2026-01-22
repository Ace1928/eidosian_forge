from collections import namedtuple
import numpy as np
from llvmlite.ir import Constant, IRBuilder
from llvmlite import ir
from numba.core import types, cgutils
from numba.core.compiler_lock import global_compiler_lock
from numba.core.caching import make_library_cache, NullCache
def _wrapper_function_type(self):
    byte_t = ir.IntType(8)
    byte_ptr_t = ir.PointerType(byte_t)
    byte_ptr_ptr_t = ir.PointerType(byte_ptr_t)
    intp_t = self.context.get_value_type(types.intp)
    intp_ptr_t = ir.PointerType(intp_t)
    fnty = ir.FunctionType(ir.VoidType(), [byte_ptr_ptr_t, intp_ptr_t, intp_ptr_t, byte_ptr_t])
    return fnty