import dis
import functools
import inspect
import sys
from types import FrameType, FunctionType
from typing import Callable, List, Optional, Tuple, TypeVar, Union, overload
from overrides.signature import ensure_signature_is_compatible
def _overrides(method: _WrappedMethod, check_signature: bool, check_at_runtime: bool) -> _WrappedMethod:
    setattr(method, '__override__', True)
    global_vars = getattr(method, '__globals__', None)
    if global_vars is None:
        global_vars = vars(sys.modules[method.__module__])
    for super_class in _get_base_classes(sys._getframe(3), global_vars):
        if hasattr(super_class, method.__name__):
            if check_at_runtime:

                @functools.wraps(method)
                def wrapper(*args, **kwargs):
                    _validate_method(method, super_class, check_signature)
                    return method(*args, **kwargs)
                return wrapper
            else:
                _validate_method(method, super_class, check_signature)
                return method
    raise TypeError(f'{method.__qualname__}: No super class method found')