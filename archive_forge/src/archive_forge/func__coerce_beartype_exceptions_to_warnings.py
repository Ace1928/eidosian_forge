import enum
import functools
import os
import traceback
import typing
import warnings
from types import ModuleType
@functools.wraps(func)
def _coerce_beartype_exceptions_to_warnings(*args, **kwargs):
    try:
        return beartyped(*args, **kwargs)
    except _roar.BeartypeCallHintParamViolation:
        warnings.warn(traceback.format_exc(), category=CallHintViolationWarning, stacklevel=2)
    return func(*args, **kwargs)