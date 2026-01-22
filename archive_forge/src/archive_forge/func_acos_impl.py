import cmath
import math
from numba.core.imputils import Registry, impl_ret_untracked
from numba.core import types, cgutils
from numba.core.typing import signature
from numba.cpython import builtins, mathimpl
from numba.core.extending import overload
def acos_impl(z):
    """cmath.acos(z)"""
    if abs(z.real) > THRES or abs(z.imag) > THRES:
        real = math.atan2(abs(z.imag), z.real)
        imag = math.copysign(math.log(math.hypot(z.real * 0.5, z.imag * 0.5)) + LN_4, -z.imag)
        return complex(real, imag)
    else:
        s1 = cmath.sqrt(complex(1.0 - z.real, -z.imag))
        s2 = cmath.sqrt(complex(1.0 + z.real, z.imag))
        real = 2.0 * math.atan2(s1.real, s2.real)
        imag = math.asinh(s2.real * s1.imag - s2.imag * s1.real)
        return complex(real, imag)