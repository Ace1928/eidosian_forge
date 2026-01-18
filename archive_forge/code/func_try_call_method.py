from functools import wraps
import inspect
import operator
from numba.core.extending import overload
from numba.core.types import ClassInstanceType
def try_call_method(cls_type, method, n_args=1):
    """
    If method is defined for cls_type, return a callable that calls this method.
    If not, return None.
    """
    if method in cls_type.jit_methods:
        arg_names = _get_args(n_args)
        template = f'\ndef func({','.join(arg_names)}):\n    return {arg_names[0]}.{method}({','.join(arg_names[1:])})\n'
        return extract_template(template, 'func')