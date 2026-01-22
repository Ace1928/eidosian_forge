import cmath
import math
from numba.core.imputils import Registry, impl_ret_untracked
from numba.core import types, cgutils
from numba.core.typing import signature
from numba.cpython import builtins, mathimpl
from numba.core.extending import overload
def asin_impl(z):
    """cmath.asin(z) = -j * cmath.asinh(z j)"""
    r = cmath.asinh(complex(-z.imag, z.real))
    return complex(r.imag, -r.real)