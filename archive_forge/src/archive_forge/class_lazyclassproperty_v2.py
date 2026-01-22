import functools
import inspect
import textwrap
import threading
import types
import warnings
from typing import TypeVar, Type, Callable, Any, Union
from inspect import signature
from functools import wraps
class lazyclassproperty_v2(Generic[T, R]):

    def __init__(self, func: Callable[[Type[T]], R]) -> None:
        self.func = func
        self._key = self.func.__name__
        self._lock = threading.RLock()

    def __get__(self, obj: Any, cls: Type[T]) -> R:
        try:
            obj_dict = obj.__dict__
            val = obj_dict.get(self._key, _NotFound)
            if val is _NotFound:
                with self._lock:
                    val = obj_dict.get(self._key, _NotFound)
                    if val is _NotFound:
                        val = self.func(cls)
                        obj_dict[self._key] = val
            return val
        except AttributeError:
            if obj is None:
                return self
            raise

    def __set__(self, obj: Any, value: Any) -> None:
        obj_dict = obj.__dict__
        obj_dict[self._key] = value

    def __delete__(self, obj: Any) -> None:
        obj_dict = obj.__dict__
        obj_dict.pop(self._key, None)