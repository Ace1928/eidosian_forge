from numba.extending import (models, register_model, type_callable,
from numba.core import types, cgutils
import warnings
from numba.core.errors import NumbaExperimentalFeatureWarning, NumbaValueError
from numpy.polynomial.polynomial import Polynomial
from contextlib import ExitStack
import numpy as np
from llvmlite import ir
@unbox(types.PolynomialType)
def unbox_polynomial(typ, obj, c):
    """
    Convert a Polynomial object to a native polynomial structure.
    """
    is_error_ptr = cgutils.alloca_once_value(c.builder, cgutils.false_bit)
    polynomial = cgutils.create_struct_proxy(typ)(c.context, c.builder)
    with ExitStack() as stack:
        natives = []
        for name in ('coef', 'domain', 'window'):
            attr = c.pyapi.object_getattr_string(obj, name)
            with cgutils.early_exit_if_null(c.builder, stack, attr):
                c.builder.store(cgutils.true_bit, is_error_ptr)
            t = getattr(typ, name)
            native = c.unbox(t, attr)
            c.pyapi.decref(attr)
            with cgutils.early_exit_if(c.builder, stack, native.is_error):
                c.builder.store(cgutils.true_bit, is_error_ptr)
            natives.append(native)
        polynomial.coef = natives[0]
        polynomial.domain = natives[1]
        polynomial.window = natives[2]
    return NativeValue(polynomial._getvalue(), is_error=c.builder.load(is_error_ptr))