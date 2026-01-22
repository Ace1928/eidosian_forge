import math
import llvmlite.ir
import numpy as np
from numba.core.extending import overload
from numba.core.imputils import impl_ret_untracked
from numba.core import typing, types, errors, lowering, cgutils
from numba.core.extending import register_jitable
from numba.np import npdatetime
from numba.cpython import cmathimpl, mathimpl, numbers
def _dispatch_func_by_name_type(context, builder, sig, args, table, user_name):
    ty = sig.args[0]
    try:
        func_name = table[ty]
    except KeyError as e:
        msg = 'No {0} function for real type {1}'.format(user_name, str(e))
        raise errors.LoweringError(msg)
    mod = builder.module
    if ty in types.complex_domain:
        out = context.make_complex(builder, ty)
        ptrargs = [cgutils.alloca_once_value(builder, arg) for arg in args]
        call_args = [out._getpointer()] + ptrargs
        call_argtys = [ty] + list(sig.args)
        call_argltys = [context.get_value_type(ty).as_pointer() for ty in call_argtys]
        fnty = llvmlite.ir.FunctionType(llvmlite.ir.VoidType(), call_argltys)
        fn = cgutils.get_or_insert_function(mod, fnty, func_name)
        builder.call(fn, call_args)
        retval = builder.load(call_args[0])
    else:
        argtypes = [context.get_argument_type(aty) for aty in sig.args]
        restype = context.get_argument_type(sig.return_type)
        fnty = llvmlite.ir.FunctionType(restype, argtypes)
        fn = cgutils.insert_pure_function(mod, fnty, name=func_name)
        retval = context.call_external_function(builder, fn, sig.args, args)
    return retval