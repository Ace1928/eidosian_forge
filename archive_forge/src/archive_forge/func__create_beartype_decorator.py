import enum
import functools
import os
import traceback
import typing
import warnings
from types import ModuleType
def _create_beartype_decorator(runtime_check_state: RuntimeTypeCheckState):
    if runtime_check_state == RuntimeTypeCheckState.DISABLED:
        return _no_op_decorator
    if _beartype_lib is None:
        return _no_op_decorator
    assert isinstance(_beartype_lib, ModuleType)
    if runtime_check_state == RuntimeTypeCheckState.ERRORS:
        return _beartype_lib.beartype

    def beartype(func):
        """Warn on type hint violation."""
        if 'return' in func.__annotations__:
            return_type = func.__annotations__['return']
            del func.__annotations__['return']
            beartyped = _beartype_lib.beartype(func)
            func.__annotations__['return'] = return_type
        else:
            beartyped = _beartype_lib.beartype(func)

        @functools.wraps(func)
        def _coerce_beartype_exceptions_to_warnings(*args, **kwargs):
            try:
                return beartyped(*args, **kwargs)
            except _roar.BeartypeCallHintParamViolation:
                warnings.warn(traceback.format_exc(), category=CallHintViolationWarning, stacklevel=2)
            return func(*args, **kwargs)
        return _coerce_beartype_exceptions_to_warnings
    return beartype