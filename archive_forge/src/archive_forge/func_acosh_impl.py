import cmath
import math
from numba.core.imputils import Registry, impl_ret_untracked
from numba.core import types, cgutils
from numba.core.typing import signature
from numba.cpython import builtins, mathimpl
from numba.core.extending import overload
def acosh_impl(z):
    """cmath.acosh(z)"""
    if abs(z.real) > THRES or abs(z.imag) > THRES:
        real = math.log(math.hypot(z.real * 0.5, z.imag * 0.5)) + LN_4
        imag = math.atan2(z.imag, z.real)
        return complex(real, imag)
    else:
        s1 = cmath.sqrt(complex(z.real - 1.0, z.imag))
        s2 = cmath.sqrt(complex(z.real + 1.0, z.imag))
        real = math.asinh(s1.real * s2.real + s1.imag * s2.imag)
        imag = 2.0 * math.atan2(s1.imag, s2.real)
        return complex(real, imag)