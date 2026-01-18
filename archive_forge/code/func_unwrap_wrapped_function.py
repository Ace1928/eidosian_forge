from __future__ import annotations as _annotations
from collections import deque
from dataclasses import dataclass, field
from functools import cached_property, partial, partialmethod
from inspect import Parameter, Signature, isdatadescriptor, ismethoddescriptor, signature
from itertools import islice
from typing import TYPE_CHECKING, Any, Callable, ClassVar, Generic, Iterable, TypeVar, Union
from pydantic_core import PydanticUndefined, core_schema
from typing_extensions import Literal, TypeAlias, is_typeddict
from ..errors import PydanticUserError
from ._core_utils import get_type_ref
from ._internal_dataclass import slots_true
from ._typing_extra import get_function_type_hints
def unwrap_wrapped_function(func: Any, *, unwrap_partial: bool=True, unwrap_class_static_method: bool=True) -> Any:
    """Recursively unwraps a wrapped function until the underlying function is reached.
    This handles property, functools.partial, functools.partialmethod, staticmethod and classmethod.

    Args:
        func: The function to unwrap.
        unwrap_partial: If True (default), unwrap partial and partialmethod decorators, otherwise don't.
            decorators.
        unwrap_class_static_method: If True (default), also unwrap classmethod and staticmethod
            decorators. If False, only unwrap partial and partialmethod decorators.

    Returns:
        The underlying function of the wrapped function.
    """
    all: set[Any] = {property, cached_property}
    if unwrap_partial:
        all.update({partial, partialmethod})
    if unwrap_class_static_method:
        all.update({staticmethod, classmethod})
    while isinstance(func, tuple(all)):
        if unwrap_class_static_method and isinstance(func, (classmethod, staticmethod)):
            func = func.__func__
        elif isinstance(func, (partial, partialmethod)):
            func = func.func
        elif isinstance(func, property):
            func = func.fget
        else:
            assert isinstance(func, cached_property)
            func = func.func
    return func