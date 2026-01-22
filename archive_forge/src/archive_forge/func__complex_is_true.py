import math
import llvmlite.ir
import numpy as np
from numba.core.extending import overload
from numba.core.imputils import impl_ret_untracked
from numba.core import typing, types, errors, lowering, cgutils
from numba.core.extending import register_jitable
from numba.np import npdatetime
from numba.cpython import cmathimpl, mathimpl, numbers
def _complex_is_true(context, builder, ty, val):
    complex_val = context.make_complex(builder, ty, value=val)
    re_true = cgutils.is_true(builder, complex_val.real)
    im_true = cgutils.is_true(builder, complex_val.imag)
    return builder.or_(re_true, im_true)