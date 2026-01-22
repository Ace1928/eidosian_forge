import functools
import inspect
import logging
import os
from typing import TYPE_CHECKING, Any, Callable, Dict, Generic, Optional, Type, TypeVar
from lightning_utilities.core.imports import RequirementCache
from torch import nn
from typing_extensions import Concatenate, ParamSpec
import pytorch_lightning as pl
class _restricted_classmethod_impl(Generic[_T, _P, _R_co]):
    """Drop-in replacement for @classmethod, but raises an exception when the decorated method is called on an instance
    instead of a class type."""

    def __init__(self, method: Callable[Concatenate[Type[_T], _P], _R_co]) -> None:
        self.method = method

    def __get__(self, instance: Optional[_T], cls: Type[_T]) -> Callable[_P, _R_co]:

        @functools.wraps(self.method)
        def wrapper(*args: Any, **kwargs: Any) -> _R_co:
            is_scripting = any((os.path.join('torch', 'jit') in frameinfo.filename for frameinfo in inspect.stack()))
            if instance is not None and (not is_scripting):
                raise TypeError(f'The classmethod `{cls.__name__}.{self.method.__name__}` cannot be called on an instance. Please call it on the class type and make sure the return value is used.')
            return self.method(cls, *args, **kwargs)
        return wrapper