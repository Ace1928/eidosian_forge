import inspect
from inspect import Parameter
from types import FunctionType
from typing import Callable, Dict, Optional, Tuple, Type, TypeVar, Union, get_type_hints
from .typing_utils import get_args, issubtype
def _is_same_module(callable1: _WrappedMethod, callable2: _WrappedMethod2) -> bool:
    mod1 = callable1.__module__.split('.')[0]
    mod2 = getattr(callable2, '__module__', None)
    if mod2 is None:
        return False
    mod2 = mod2.split('.')[0]
    return mod1 == mod2