import collections
import contextlib
import inspect
import functools
from enum import Enum
from numba.core import typing, types, utils, cgutils
from numba.core.typing.templates import BaseRegistryLoader
def user_function(fndesc, libs):
    """
    A wrapper inserting code calling Numba-compiled *fndesc*.
    """

    def imp(context, builder, sig, args):
        func = context.declare_function(builder.module, fndesc)
        status, retval = context.call_conv.call_function(builder, func, fndesc.restype, fndesc.argtypes, args)
        with cgutils.if_unlikely(builder, status.is_error):
            context.call_conv.return_status_propagate(builder, status)
        assert sig.return_type == fndesc.restype
        retval = fix_returning_optional(context, builder, sig, status, retval)
        if retval.type != context.get_value_type(sig.return_type):
            msg = 'function returned {0} but expect {1}'
            raise TypeError(msg.format(retval.type, sig.return_type))
        return impl_ret_new_ref(context, builder, fndesc.restype, retval)
    imp.signature = fndesc.argtypes
    imp.libs = tuple(libs)
    return imp