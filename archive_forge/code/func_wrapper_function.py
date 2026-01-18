from __future__ import annotations as _annotations
import functools
from typing import TYPE_CHECKING, Any, Callable, TypeVar, overload
from ._internal import _validate_call
@functools.wraps(function)
def wrapper_function(*args, **kwargs):
    return validate_call_wrapper(*args, **kwargs)