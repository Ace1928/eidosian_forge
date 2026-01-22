from functools import wraps, partial
from llvmlite import ir
from numba.core import types, cgutils
from numba.core.decorators import njit
from numba.core.pythonapi import box, unbox, NativeValue
from numba.core.typing.typeof import typeof_impl
from numba.experimental.jitclass import _box
def _generate_method(name, func):
    """
    Generate a wrapper for calling a method.  Note the wrapper will only
    accept positional arguments.
    """
    source = _method_code_template.format(method=name)
    glbls = {}
    exec(source, glbls)
    method = njit(glbls['method'])

    @wraps(func)
    def wrapper(*args, **kwargs):
        return method(*args, **kwargs)
    return wrapper