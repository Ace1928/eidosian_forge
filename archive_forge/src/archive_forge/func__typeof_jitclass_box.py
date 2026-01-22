from functools import wraps, partial
from llvmlite import ir
from numba.core import types, cgutils
from numba.core.decorators import njit
from numba.core.pythonapi import box, unbox, NativeValue
from numba.core.typing.typeof import typeof_impl
from numba.experimental.jitclass import _box
@typeof_impl.register(_box.Box)
def _typeof_jitclass_box(val, c):
    return getattr(type(val), '_numba_type_')